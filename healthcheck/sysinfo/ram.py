"""Memory size, this job's limit, hugepages and NUMA nodes; adapted from srygaard/mlps shared/mlps_shared/sysinfo.py."""

from pathlib import Path

from healthcheck.util.file import read_sys


def meminfo(key: str) -> int | None:
    """A /proc/meminfo value as listed: kB for sizes, a count for HugePages_*."""
    try:
        with open("/proc/meminfo") as proc_meminfo:
            for line in proc_meminfo:
                if line.startswith(f"{key}:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return None


def total_bytes() -> int | None:
    kb = meminfo("MemTotal")
    return kb * 1024 if kb is not None else None


def allocated_bytes() -> int | None:
    """Memory hard limit from cgroup, None if unconstrained or unavailable."""
    raw = read_sys("/sys/fs/cgroup/memory.max")  # cgroup v2
    if raw and raw != "max":
        return int(raw)
    raw = read_sys("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1
    total = total_bytes() or 0
    if raw and total and int(raw) < total:
        return int(raw)
    return None


def numa_nodes() -> int:
    return len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
