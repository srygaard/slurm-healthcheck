"""Query the cluster and submit its Slurm jobs."""

import json
import os
import re
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self, TypedDict, cast

from healthcheck.util.cmd import run
from healthcheck.util.hostlist import Hostlist

_SBATCH = re.compile(r"^#SBATCH\s+--([\w-]+)=(\S+)", re.MULTILINE)


def directives(text: str) -> dict[str, str]:
    """`#SBATCH --option=value` lines. The value ends at the first space, so a trailing comment is left out.

    >>> directives("#SBATCH --gpus-per-node=b200:8\\n#SBATCH --time=00:30:00\\n")
    {'gpus-per-node': 'b200:8', 'time': '00:30:00'}
    """
    return dict(_SBATCH.findall(text))


class Gres(NamedTuple):
    """A `--gpus-per-node` value: the chip name and how many of them. name is None for a bare count."""

    name: str | None
    count: int

    def __str__(self) -> str:
        return f"{self.name}:{self.count}" if self.name else str(self.count)


@dataclass(frozen=True)
class JobTemplate:
    """A Slurm job script. Directives are parsed once from the file text."""

    text: str
    directives: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directives", directives(self.text))

    @classmethod
    def from_sh(cls, path: Path) -> Self:
        return cls(Path(path).read_text())

    @classmethod
    def load(cls, directory: Path | None = None) -> Self:
        """slurm/templates/<cluster>.sh. Raises ValueError when $CC_CLUSTER has no such file."""
        name = detect_cluster()
        root = Path(__file__).resolve().parents[1] / "slurm" / "templates" if directory is None else directory
        path = root / f"{name}.sh" if name else None
        if path is None or not path.is_file():
            raise ValueError(f"no job template for {name or '?'}")
        return cls.from_sh(path)

    @property
    def gres(self) -> Gres | None:
        """--gpus-per-node. None when that line is absent.

        >>> JobTemplate("#SBATCH --gpus-per-node=b200:8\\n").gres
        Gres(name='b200', count=8)
        >>> JobTemplate("#SBATCH --gpus-per-node=h100:4\\n").gres
        Gres(name='h100', count=4)
        >>> JobTemplate("#SBATCH --gpus-per-node=4\\n").gres
        Gres(name=None, count=4)
        >>> JobTemplate("").gres is None
        True
        """
        raw = self.directives.get("gpus-per-node")
        if not raw:
            return None
        # Unsupported: gpu:h100:4, gpu:4, h100:4(S:0-1), a comma list (gpu:h100:4,mps:100),
        # and a GPU requested only as --gres=gpu:h100:4, --gpus=h100:4, -G h100:4,
        # --gpus-per-task, or --gpus-per-socket. #SBATCH --gpus-per-node h100:4 (no "=") is also missed.
        name, sep, count = raw.partition(":")
        token = count if sep else name
        try:
            parsed = int(token)
        except ValueError:
            raise ValueError(f"invalid --gpus-per-node count in {raw}") from None
        if not sep:
            return Gres(None, parsed)
        return Gres(name, parsed)

    @property
    def time(self) -> str | None:
        return self.directives.get("time")

    @property
    def account(self) -> str | None:
        """None when the script has no --account line, as in mila.sh."""
        return self.directives.get("account")

    @property
    def reservation(self) -> str | None:
        """--reservation. None when the script has no reservation line."""
        return self.directives.get("reservation")

    @property
    def constraint(self) -> list[str]:
        """--constraint features joined by &. Raises ValueError unless the expression is a plain AND.

        >>> JobTemplate("#SBATCH --constraint=h100&nvlink\\n").constraint
        ['h100', 'nvlink']
        >>> JobTemplate("").constraint
        []
        """
        raw = self.directives.get("constraint", "")
        if re.search(r"[|\[\]*()]", raw):
            raise ValueError(f"Only AND constraints (a&b) can select nodes, not {raw}")
        return raw.split("&") if raw else []


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


@dataclass(frozen=True)
class SqueueJob:
    job_id: int
    state: JobState
    start_time: datetime | None = None

    def __str__(self) -> str:
        time_info = ""
        if self.start_time:
            prefix = "estimated start" if self.state == "PENDING" else "started"
            time_info = f" ({prefix} {self.start_time:%Y-%m-%dT%H:%M:%S})"
        return f"{self.job_id} {self.state}{time_info}"


class SqueueResult(NamedTuple):
    jobs: list[SqueueJob]
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
    # cwd: job_submit.lua may refuse jobs submitted from /home; a missing cwd is ExitCode.NOT_FOUND
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


def _start_time(value: Any) -> datetime | None:
    """Parse start_time from squeue --json into a datetime or None."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, dict):
        if not value.get("set") or value.get("infinite"):
            return None
        value = value.get("number")
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value).astimezone()
    if isinstance(value, str) and value and value != "N/A":
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.astimezone()
        except ValueError:
            return None
    return None


def squeue(options: Sequence[str]) -> SqueueResult:
    """The queued jobs that options (-j, --me, -n ...) select, with their states and start times."""
    result = run(["squeue", "--json", *options], timeout=None)
    jobs = [
        SqueueJob(
            job_id=job["job_id"],
            state=_state(job["job_state"]),
            start_time=_start_time(job.get("start_time")),
        )
        for job in _jobs(result.stdout)
    ]
    return SqueueResult(jobs, result.returncode, result.stderr)


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
