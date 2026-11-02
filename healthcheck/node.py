"""Per-node entry point, one srun task per node: `python -m healthcheck.node {watch_xid,local,finalize}`.

watch_xid is the overlapping Xid thread and stays up for the job. local and finalize are phases A and C.
Each node writes only its own <run>/nodes/<host>.json and <run>/numa/<host>.json.
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from healthcheck import cpu_load, dcgm, local_tmp, logs, share, sysinfo, xid
from healthcheck.check import HealthCheckBase, step
from healthcheck.config import cfg
from healthcheck.util.file import read_json, write_json
from healthcheck.util.status import ok

# Node steps in report order. logs is phase C; local() runs the rest.
STEPS: dict[str, type[HealthCheckBase]] = {
    "sysinfo": sysinfo.HealthCheck,
    "scratch": share.HealthCheck,
    "project": share.HealthCheck,
    "local_tmp": local_tmp.HealthCheck,
    "dcgm": dcgm.HealthCheck,
    "cpu_load": cpu_load.HealthCheck,
    "logs": logs.HealthCheck,
}


def _check(name: str, host: str, gpus_done: threading.Event) -> HealthCheckBase:
    """An instance of one phase A step. logs is built in finalize, where its inputs exist."""
    if name == "sysinfo":
        return STEPS[name](host, cfg.nic.link_layer)
    if name == "scratch":
        return STEPS[name](cfg.paths.base, cfg.scratch, host)
    if name == "project":
        return STEPS[name](cfg.paths.project, cfg.project, host)
    if name == "local_tmp":
        return STEPS[name](cfg.local_tmp, host)
    if name == "dcgm":
        return STEPS[name](cfg.dcgm, cfg.paths.dcgm_logs(host))
    if name == "cpu_load":
        return STEPS[name](cfg.cpu_load, gpus_done, sorted(os.sched_getaffinity(0)))
    raise KeyError(name)


def local() -> None:
    """Steps 1, 2a, 2b, 2c and 3a concurrently; the job's time limit caps them."""
    host = cfg.host.name
    path = cfg.paths.node(host)
    out: dict[str, Any] = {
        "host": host,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    # cpu_load holds its CPU load until dcgm returns
    gpus_done = threading.Event()
    checks = {
        name: _check(name, host, gpus_done)
        for name in STEPS
        if name != "logs"
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {name: pool.submit(step, check) for name, check in checks.items()}
        futures["dcgm"].add_done_callback(lambda _: gpus_done.set())
        for name, future in futures.items():
            out[name] = future.result()
            print(f"{host} {name}: {out[name]['status']} in {out[name]['elapsed_s']}s")
    # sysinfo returns the NUMA layout for its own file; the node's results are written first, so they survive a failure
    numa_layout = out["sysinfo"].get("data", {}).pop(sysinfo.NUMA_LAYOUT, None) if ok(out["sysinfo"]) else None
    write_json(path, out)
    if numa_layout is not None:
        write_json(cfg.paths.numa(host), numa_layout)


def watch_xid() -> None:
    """Record NVIDIA Xid errors, the GPU driver's fault codes, until the batch step touches the stop file after NCCL."""
    if not cfg.logs.xid_watch:
        return
    watcher = xid.Watcher()
    watcher.start()
    stop_xid = cfg.paths.xid_stop()
    while not stop_xid.exists():
        time.sleep(1)
    watcher.stop()


def finalize() -> None:
    """Step 4, merged into the node's JSON from phase A."""
    host = cfg.host.name
    path = cfg.paths.node(host)
    out = read_json(path) or {"host": host, "job_id": os.environ.get("SLURM_JOB_ID")}
    before = (out.get("sysinfo") or {}).get("data", {}).get("counters")
    since = float(os.environ.get("SLURM_JOB_START_TIME", time.time() - 600))
    out["logs"] = step(STEPS["logs"](cfg.logs, cfg.paths, host, since, before))
    out["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(path, out)
    print(f"{host} logs: {out['logs']['status']} in {out['logs']['elapsed_s']}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["watch_xid", "local", "finalize"])
    return parser.parse_args()


def main() -> None:
    phase = parse_args().phase
    if phase == "watch_xid":
        watch_xid()
    elif phase == "local":
        local()
    elif phase == "finalize":
        finalize()


if __name__ == "__main__":
    main()
