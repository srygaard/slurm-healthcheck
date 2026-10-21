"""Query the cluster and its Slurm jobs."""

import os
import socket
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, TypedDict, cast, get_args

from healthcheck.util.hostlist import Hostlist

type Cluster = Literal["mila", "fir", "narval", "nibi", "rorqual", "trillium", "unknown"]
type JobState = Literal[
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PENDING",
    "PREEMPTED",
    "REQUEUED",
    "RESIZING",
    "REVOKED",
    "RUNNING",
    "SUSPENDED",
    "TIMEOUT",
]


class SacctJob(TypedDict):
    state: JobState
    elapsed_s: int
    nodes: Hostlist  # empty if the job never started


def detect_cluster() -> Cluster:
    cluster = os.environ.get("CC_CLUSTER", "").strip().lower()
    if cluster in get_args(Cluster.__value__):
        return cast(Cluster, cluster)
    if socket.getfqdn().lower().endswith("mila.quebec"):
        return "mila"
    return "unknown"


def wait_for(job_ids: list[int], interval: float = 15) -> None:
    """Poll squeue until none of the jobs remain."""
    ids = ",".join(map(str, job_ids))
    while True:
        out = subprocess.run(["squeue", "-h", "-j", ids, "-o", "%i %T"], capture_output=True, text=True)
        if out.returncode and "Invalid job id" not in out.stderr:
            time.sleep(interval)  # transient slurmctld error: retry
            continue
        states = Counter(line.split()[1] for line in out.stdout.splitlines() if line.strip())
        done = len(job_ids) - sum(states.values())
        print(
            f"\r[{datetime.now():%H:%M:%S}] {done}/{len(job_ids)} done  "
            + "  ".join(f"{state.lower()} {count}" for state, count in sorted(states.items()))
            + " " * 10,
            end="",
            flush=True,
        )
        if not states:
            print()
            return
        time.sleep(interval)


def sacct_states(job_ids: Sequence[int]) -> dict[int, SacctJob]:
    if not job_ids:
        return {}
    out = subprocess.run(
        ["sacct", "-n", "-P", "-X", "-j", ",".join(map(str, job_ids)), "-o", "JobID,State,ElapsedRaw,NodeList"],
        capture_output=True,
        text=True,
    )
    states: dict[int, SacctJob] = {}
    for line in out.stdout.splitlines():
        job_id, state, elapsed, nodelist = line.split("|")
        if job_id.isdigit():
            states[int(job_id)] = {
                "state": cast(JobState, state.split()[0]),
                "elapsed_s": int(elapsed),
                "nodes": Hostlist() if nodelist == "None assigned" else Hostlist.expand(nodelist),
            }
    return states
