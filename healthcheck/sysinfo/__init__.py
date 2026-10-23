"""Step 1: node configuration (CPU, RAM, GPU, NIC, software) for threshold and homogeneity checks."""

import platform

from healthcheck.sysinfo import gpu, job, nic, software, topology
from healthcheck.util.file import JsonDict, read_sys
from healthcheck.util.nvml import session

# The key of the NUMA layout table in collect()'s result; node.local() moves it to the node's own numa/<host>.json
NUMA_LAYOUT = "numa_layout"


def collect(host: str, link_layer: str) -> JsonDict:
    """link_layer is the one NCCL's NICs use: the NUMA layout lists those HCAs apart from the rest."""
    total = topology.total_bytes()
    placements = topology.placements()
    nodes = topology.numa_nodes(placements)
    with session():  # one NVML init for devices(), info() and counters(): slow without persistence mode
        gpus = gpu.devices()
        gpu_info = gpu.info()
        counters = gpu.counters()
    hcas = nic.hcas()
    return {
        "host": host,
        "kernel": platform.release(),
        "os_release": software.os_release(),
        "cmdline": read_sys("/proc/cmdline"),
        "cpu": [socket.to_json() for socket in topology.sockets(placements)],
        "ram_gib": round(total / 2**30, 1) if total else None,
        "allocation": job.allocation()._asdict(),
        "numa_nodes": [node.to_json() for node in nodes],
        "hugepages": topology.meminfo("HugePages_Total"),
        "gpu": {"count": len(gpus), **gpu_info, "devices": [device.to_json() for device in gpus]},
        "counters": [snapshot.to_json() for snapshot in counters],
        "nic": {"ib": [hca.to_json() for hca in hcas], **nic.collect()},
        "software": software.collect(),
        NUMA_LAYOUT: topology.numa_layout(nodes, gpus, hcas, link_layer),
    }
