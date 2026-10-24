"""Query the cluster, and submit and follow its Slurm jobs."""

import json
import os
import socket
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict, cast

from healthcheck.util.cmd import run
from healthcheck.util.hostlist import Hostlist

# The JOB STATE CODES of `man squeue` (Slurm 25.11); sacct reports the same names
type JobState = Literal[
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "COMPLETING",
    "CONFIGURING",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PENDING",
    "PREEMPTED",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RESV_DEL_HOLD",
    "REVOKED",
    "RUNNING",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
    "TIMEOUT",
]


class SbatchResult(NamedTuple):
    job_id: int | None  # None if the submission failed
    returncode: int
    stdout: str
    stderr: str


class SqueueResult(NamedTuple):
    jobs: dict[int, JobState]
    returncode: int
    stderr: str


class SacctJob(TypedDict):
    state: JobState
    elapsed_s: int
    nodes: Hostlist  # empty if the job never started


def detect_cluster() -> str | None:
    """$CC_CLUSTER on the Alliance clusters, "mila" in the mila.quebec domain, None elsewhere."""
    if cluster := os.environ.get("CC_CLUSTER", "").strip().lower():
        return cluster
    if socket.getfqdn().lower().endswith("mila.quebec"):
        return "mila"
    return None


def sbatch(script: str, options: Sequence[str], cwd: Path, env: Mapping[str, str]) -> SbatchResult:
    """
    Submit a job script to SLURM; options override the script's #SBATCH directives. The job starts with this
    environment plus env, before the script's first line: --export=ALL, unlike --export=NAME=value, takes any value.
    """
    # cwd: job_submit.lua may refuse jobs submitted from /home; a missing cwd also gives 127
    command = ["sbatch", "--parsable", "--export=ALL", *options]
    result = run(command, timeout=None, input=script, cwd=cwd, env=os.environ | env)
    try:
        job_id = int(result.stdout.split(";")[0])
    except ValueError:
        job_id = None
    return SbatchResult(job_id, result.returncode, result.stdout, result.stderr)


def _jobs(text: str) -> list[dict[str, Any]]:
    """The "jobs" of squeue or sacct --json; [] for no output, as for an invalid job id."""
    return json.loads(text)["jobs"] if text.strip() else []


def _state(value: list[str] | str) -> JobState:
    """
    The base state: data_parser v0.0.40+ lists it before its flags; older ones give "CANCELLED by 123".

    >>> _state(["PENDING", "REQUEUED"]), _state("CANCELLED by 3157949")
    ('PENDING', 'CANCELLED')
    """
    return cast(JobState, value[0] if isinstance(value, list) else value.split()[0])


def squeue(options: Sequence[str]) -> SqueueResult:
    """The queued jobs that options (-j, --me, -n ...) select, with their states."""
    result = run(["squeue", "--json", *options], timeout=None)
    jobs = {job["job_id"]: _state(job["job_state"]) for job in _jobs(result.stdout)}
    return SqueueResult(jobs, result.returncode, result.stderr)


def wait_for(job_ids: list[int], interval: float = 15) -> None:
    """Poll squeue until none of the jobs remain."""
    ids = ",".join(map(str, job_ids))
    while True:
        result = squeue(["-j", ids])
        if result.returncode and "Invalid job id" not in result.stderr:
            time.sleep(interval)  # transient slurmctld error: retry
            continue
        states = Counter(result.jobs.values())
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


def my_queued(name: str) -> list[int]:
    """My pending or running jobs named name; [] if squeue fails or isn't installed."""
    return sorted(squeue(["--me", "-n", name]).jobs)


def with_dependency(dependency: str | None, job_ids: Sequence[int]) -> str | None:
    """
    An sbatch --dependency that also waits for job_ids to succeed; "," means all of them must hold.

    >>> with_dependency("afterany:7", [12, 13])
    'afterany:7,afterok:12:13'
    """
    after = f"afterok:{':'.join(map(str, job_ids))}" if job_ids else None
    return ",".join(part for part in (dependency, after) if part) or None


def sacct_states(job_ids: Sequence[int]) -> dict[int, SacctJob]:
    """Each job's state, elapsed time and nodes, from `sacct -X --json`."""
    if not job_ids:
        return {}
    out = run(["sacct", "-X", "--json", "-j", ",".join(map(str, job_ids))], timeout=None)
    return {
        job["job_id"]: {
            "state": _state(job["state"]["current"]),
            "elapsed_s": job["time"]["elapsed"],
            "nodes": Hostlist() if job["nodes"] == "None assigned" else Hostlist.expand(job["nodes"]),
        }
        for job in _jobs(out.stdout)
    }
