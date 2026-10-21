"""Step 1: node configuration (CPU, RAM, GPU, NIC, software) for threshold and homogeneity checks."""

import platform

from healthcheck.sysinfo import cpu, gpu, nic, ram, software
from healthcheck.util.file import JsonDict, read_sys


def collect(host: str) -> JsonDict:
    total = ram.total_bytes()
    allocated = ram.allocated_bytes()
    return {
        "host": host,
        "kernel": platform.release(),
        "os_release": software.os_release(),
        "cmdline": read_sys("/proc/cmdline"),
        "cpu": cpu.collect(),
        "ram_gib": round(total / 2**30, 1) if total else None,
        "ram_allocated_gib": round(allocated / 2**30, 1) if allocated else None,
        "numa_nodes": ram.numa_nodes(),
        "hugepages": ram.meminfo("HugePages_Total"),
        "gpu": gpu.info(),
        "counters": gpu.counters(),
        "nic": nic.collect(),
        "software": software.collect(),
    }
