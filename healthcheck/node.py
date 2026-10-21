"""Per-node entry point, one srun task per node: `python -m healthcheck.node {local,watch,finalize}`.

Each node writes only its own <run>/nodes/<host>.json.
"""

import argparse
import json
import os
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pynvml

from healthcheck import dcgm, local_tmp, logs, scratch, sysinfo
from healthcheck.config import cfg
from healthcheck.util.file import JsonDict, read_json, write_json
from healthcheck.util.nvml import session


def step(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> JsonDict:
    """Run one check; a crash is recorded rather than propagated, so other checks still report."""
    start = time.monotonic()
    try:
        result = {"status": "ok", "data": fn(*args, **kwargs)}
    except Exception as error:  # noqa: BLE001 - any failure is a node finding
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    result["elapsed_s"] = round(time.monotonic() - start, 2)
    return result


def local() -> None:
    """Steps 1, 2a, 2b and 3a concurrently; the job's time limit caps them."""
    host = cfg.host.name
    env = read_json(cfg.paths.env) or {}
    path = cfg.paths.node(host)
    out: dict[str, Any] = {
        "host": host,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "env": env,
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "sysinfo": pool.submit(step, sysinfo.collect, host),
            "scratch": pool.submit(step, scratch.collect, cfg.paths.base, cfg.scratch, host),
            "local_tmp": pool.submit(step, local_tmp.collect, cfg.local_tmp, host),
            "dcgm": pool.submit(step, dcgm.collect, cfg.dcgm, env, cfg.paths.dcgm_logs(host)),
        }
        for name, future in futures.items():
            out[name] = future.result()
            print(f"{host} {name}: {out[name]['status']} in {out[name]['elapsed_s']}s", flush=True)
    write_json(path, out)


def watch() -> None:
    """Record NVML Xid events until the stop file appears; this stands in for the root-only kernel log's NVRM lines.

    A stop file rather than a signal: srun answers SIGTERM by killing the step, before "stopped" is written.
    """
    if not cfg.logs.xid_watch:
        return
    path = cfg.paths.xid(cfg.host.name)
    stop_file = cfg.paths.xid_stop(os.environ.get("SLURM_JOB_ID", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def on_term(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    with open(path, "a", buffering=1) as xid_log:

        def emit(**event: Any) -> None:
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            xid_log.write(json.dumps({"time": stamp, **event}) + "\n")

        with session():
            events = pynvml.nvmlEventSetCreate()
            for index in range(pynvml.nvmlDeviceGetCount()):
                try:
                    pynvml.nvmlDeviceRegisterEvents(
                        pynvml.nvmlDeviceGetHandleByIndex(index), pynvml.nvmlEventTypeXidCriticalError, events
                    )
                except pynvml.NVMLError as error:
                    emit(event="error", gpu=index, error=str(error))
            emit(event="started")
            while not stop and not stop_file.exists():
                try:
                    data = pynvml.nvmlEventSetWait_v2(events, 1000)
                except pynvml.NVMLError_Timeout:
                    continue
                except pynvml.NVMLError as error:
                    emit(event="error", error=str(error))
                    time.sleep(1)
                    continue
                emit(event="xid", gpu=pynvml.nvmlDeviceGetIndex(data.device), xid=int(data.eventData))
            emit(event="stopped")
            pynvml.nvmlEventSetFree(events)


def finalize() -> None:
    """Step 4, merged into the node's JSON from phase A."""
    host = cfg.host.name
    path = cfg.paths.node(host)
    out = read_json(path) or {"host": host, "job_id": os.environ.get("SLURM_JOB_ID")}
    before = (out.get("sysinfo") or {}).get("data", {}).get("counters")
    since = float(os.environ.get("SLURM_JOB_START_TIME", time.time() - 600))
    out["logs"] = step(logs.collect, cfg.logs, cfg.paths, host, since, before)
    out["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(path, out)
    print(f"{host} logs: {out['logs']['status']} in {out['logs']['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["local", "watch", "finalize"])
    {"local": local, "watch": watch, "finalize": finalize}[parser.parse_args().phase]()
