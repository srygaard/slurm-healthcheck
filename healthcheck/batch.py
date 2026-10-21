"""The healthcheck job's batch step: phases A to C as srun steps across the job's nodes, with NCCL run from here.

A failed phase is recorded in the node JSONs and the next one still runs.
"""

import os
import subprocess
import sys
import traceback
from datetime import datetime

from healthcheck import nccl
from healthcheck.config import cfg


def node_command(command: str, *srun_args: str) -> list[str]:
    """`python -m healthcheck.node <command>`, one task per node."""
    return ["srun", "--ntasks-per-node=1", *srun_args, sys.executable, "-m", "healthcheck.node", command]


def main() -> int:
    job_id = os.environ["SLURM_JOB_ID"]
    cfg.paths.root.mkdir(parents=True, exist_ok=True)
    os.chdir(cfg.paths.root)  # the srun steps inherit it
    print(
        f"== {datetime.now():%T} job {job_id} on {os.environ['SLURM_JOB_NODELIST']}, results in {cfg.paths.root}",
        flush=True,
    )

    # --overlap: the watcher shares CPUs and GPUs with the test steps instead of blocking them.
    # It exits at once unless logs.xid_watch is set.
    watcher = subprocess.Popen(node_command("watch", "--overlap", "--cpus-per-task=1"))

    print(f"== {datetime.now():%T} A: sysinfo, scratch, local_tmp, dcgm", flush=True)
    subprocess.run(node_command("local"), check=False)

    print(f"== {datetime.now():%T} B: nccl", flush=True)
    try:
        nccl.main()
    except Exception:  # noqa: BLE001 - the report shows the missing NCCL results
        traceback.print_exc()

    stop_file = cfg.paths.xid_stop(job_id)  # the watcher polls for it each second, see healthcheck.node.watch
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    try:
        watcher.wait(timeout=20)
    except subprocess.TimeoutExpired:
        watcher.terminate()

    print(f"== {datetime.now():%T} C: logs", flush=True)
    subprocess.run(node_command("finalize"), check=False)
    print(f"== {datetime.now():%T} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
