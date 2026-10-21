"""CPU model, core count and this job's allocation; adapted from srygaard/mlps shared/mlps_shared/sysinfo.py."""

import math
import os
import platform

from healthcheck.util.file import JsonDict, read_sys


def model() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def allocated() -> int | None:
    """CPUs available to this process (affinity, then cgroup quota), None if unconstrained."""
    affinity = os.sched_getaffinity(0)
    if len(affinity) < (os.cpu_count() or 0):
        return len(affinity)
    if (raw := read_sys("/sys/fs/cgroup/cpu.max")) and not raw.startswith("max"):
        quota, period = raw.split()
        return max(1, math.ceil(int(quota) / int(period)))
    return None


def collect() -> JsonDict:
    return {"model": model(), "cores": os.cpu_count(), "allocated": allocated()}
