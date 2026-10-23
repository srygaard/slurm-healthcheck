"""This job's share of the node: the CPUs and memory its affinity and cgroup limits allow."""

import math
import os
from typing import NamedTuple

from healthcheck.sysinfo.topology import total_bytes
from healthcheck.util.file import read_sys


class Allocation(NamedTuple):
    """None where the job is unconstrained."""

    cpus: int | None
    memory_gib: float | None


def cpus() -> int | None:
    """CPUs available to this process (affinity, then cgroup quota), None if unconstrained."""
    affinity = os.sched_getaffinity(0)
    if len(affinity) < (os.cpu_count() or 0):
        return len(affinity)
    if (raw := read_sys("/sys/fs/cgroup/cpu.max")) and not raw.startswith("max"):
        quota, period = raw.split()
        return max(1, math.ceil(int(quota) / int(period)))
    return None


def memory_bytes() -> int | None:
    """Memory hard limit from cgroup, None if unconstrained or unavailable."""
    raw = read_sys("/sys/fs/cgroup/memory.max")  # cgroup v2
    if raw and raw != "max":
        return int(raw)
    raw = read_sys("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1
    total = total_bytes() or 0
    if raw and total and int(raw) < total:
        return int(raw)
    return None


def allocation() -> Allocation:
    memory = memory_bytes()
    return Allocation(cpus=cpus(), memory_gib=round(memory / 2**30, 1) if memory else None)
