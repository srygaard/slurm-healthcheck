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
    print(f"== {datetime.now():%T} job {job_id} on {os.environ['SLURM_JOB_NODELIST']}, results in {cfg.paths.root}")

    # --overlap: the watcher shares CPUs and GPUs with the test steps instead of blocking them.
    # It exits at once unless logs.xid_watch is set.
    watcher = subprocess.Popen(
        ["srun", "--ntasks-per-node=1", "--overlap", "--cpus-per-task=1", sys.executable, "-m", "healthcheck.xid"]
    )

    print(f"== {datetime.now():%T} A: sysinfo, $SCRATCH, $PROJECT, local_tmp, dcgm")
    subprocess.run(node_command("local"), check=False)

    print(f"== {datetime.now():%T} B: nccl")
    code = 0
    try:
        # enabled, then available, then collect. Disabled or a missing tool writes the group file and does not launch.
        nccl.start(nccl.Check(cfg.nccl, cfg.paths, cfg.gpu.count))
    except Exception:  # noqa: BLE001 - the report shows the missing NCCL results
        traceback.print_exc()
        code = 1

    stop_file = cfg.paths.xid_stop(job_id)  # the watcher polls for it each second, see healthcheck.xid.watch
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    try:
        watcher.wait(timeout=20)
    except subprocess.TimeoutExpired:
        watcher.terminate()

    print(f"== {datetime.now():%T} C: logs")
    subprocess.run(node_command("finalize"), check=False)
    print(f"== {datetime.now():%T} done")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
