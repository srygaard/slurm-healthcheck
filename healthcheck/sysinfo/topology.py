"""CPU sockets, NUMA nodes and memory from one walk over sysfs's CPUs, and the NUMA layout of the node's devices.

Adapted from srygaard/mlps shared/mlps_shared/sysinfo.py.
"""

import platform
from dataclasses import dataclass
from pathlib import Path
from string import ascii_letters
from typing import Any, NamedTuple, Self

from healthcheck.sysinfo.gpu import Gpu
from healthcheck.sysinfo.nic import Hca
from healthcheck.util.device import Device
from healthcheck.util.file import JsonDict, read_sys


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


def models(cpuinfo: Path = Path("/proc/cpuinfo")) -> dict[int | None, str]:
    """
    Each socket's model name, from the "physical id" and "model name" of /proc/cpuinfo's per-CPU blocks; under None,
    the first block's, for the VMs and containers that list no physical id.
    """
    out: dict[int | None, str] = {}
    for block in (read_sys(cpuinfo) or "").split("\n\n"):
        pairs = (line.partition(":") for line in block.splitlines())  # "model name\t: AMD EPYC 9454 ..."
        fields = {key.strip(): value.strip() for key, _, value in pairs}
        if "model name" in fields:
            out.setdefault(None, fields["model name"])
            if "physical id" in fields:
                out.setdefault(int(fields["physical id"]), fields["model name"])
    return out


class Placement(NamedTuple):
    """Where one online CPU sits."""

    cpu: int
    socket: int
    numa_node: int | None


def _number(path: Path) -> int:
    """
    The number in a sysfs cpu or node directory's name.

    >>> _number(Path("/sys/devices/system/node/node12"))
    12
    """
    return int(path.name.lstrip(ascii_letters))


def _placement(path: Path) -> Placement | None:
    """From /sys/devices/system/cpu/cpu<N>; None for an offline CPU, which has no topology."""
    if (package := read_sys(path / "topology" / "physical_package_id")) is None:
        return None
    node = next(path.glob("node[0-9]*"), None)
    return Placement(_number(path), int(package), _number(node) if node else None)


def placements(root: Path = Path("/sys")) -> list[Placement]:
    """Each online CPU's socket and NUMA node; sockets() and numa_nodes() group these, so they always agree."""
    paths = (root / "devices" / "system" / "cpu").glob("cpu[0-9]*")
    return sorted(placement for path in paths if (placement := _placement(path)))


def _cpulist(cpus: list[int]) -> str | None:
    """
    The kernel's cpulist form; None for no CPUs.

    >>> _cpulist([0, 1, 2, 3, 8, 10, 11])
    '0-3,8,10-11'
    """
    ranges: list[list[int]] = []
    for cpu in sorted(cpus):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{first}-{last}" if first != last else f"{first}" for first, last in ranges) or None


@dataclass(frozen=True, kw_only=True)
class Cpu(Device):
    """One socket; name is its model. numa_node is None if the socket spans several NUMA nodes (AMD NPS2, NPS4)."""

    socket: int
    cpus: str | None  # "0-15,32-47"
    cores: int  # logical CPUs online
    numa_nodes: list[int]

    @classmethod
    def from_placements(cls, socket: int, placements: list[Placement], model: str | None) -> Self:
        """From the placements of every CPU; it keeps the socket's own."""
        mine = [cpu for cpu in placements if cpu.socket == socket]
        numa_nodes = sorted({cpu.numa_node for cpu in mine if cpu.numa_node is not None})
        return cls(
            name=model,
            numa_node=numa_nodes[0] if len(numa_nodes) == 1 else None,
            socket=socket,
            cpus=_cpulist([cpu.cpu for cpu in mine]),
            cores=len(mine),
            numa_nodes=numa_nodes,
        )


def sockets(placements: list[Placement], cpuinfo: Path = Path("/proc/cpuinfo")) -> list[Cpu]:
    names = models(cpuinfo)
    return [
        Cpu.from_placements(socket, placements, names.get(socket) or names.get(None) or platform.processor() or None)
        for socket in sorted({cpu.socket for cpu in placements})
    ]


@dataclass(frozen=True, kw_only=True)
class NumaNode(Device):
    """A NUMA node, e.g. node3: its socket, CPUs ("36-47") and memory. A memory-only node (CXL, HBM) has no CPUs."""

    socket: int | None
    cpus: str | None
    mem_gib: float | None

    @classmethod
    def from_sysfs(cls, path: Path, placements: list[Placement]) -> Self:
        """From its /sys/devices/system/node/node<N> directory and the placements of every CPU; it keeps its own."""
        numa_node = _number(path)
        mine = [cpu for cpu in placements if cpu.numa_node == numa_node]
        sockets = {cpu.socket for cpu in mine}
        mem_gib = None
        for line in (read_sys(path / "meminfo") or "").splitlines():
            if "MemTotal:" in line:  # "Node 0 MemTotal:  296955060 kB"
                mem_gib = round(int(line.split()[3]) / 2**20, 1)
        return cls(
            name=path.name,
            numa_node=numa_node,
            socket=sockets.pop() if len(sockets) == 1 else None,
            cpus=_cpulist([cpu.cpu for cpu in mine]),
            mem_gib=mem_gib,
        )


def numa_nodes(placements: list[Placement], root: Path = Path("/sys")) -> list[NumaNode]:
    paths = (root / "devices" / "system" / "node").glob("node[0-9]*")
    return [NumaNode.from_sysfs(path, placements) for path in sorted(paths, key=_number)]


def numa_layout(nodes: list[NumaNode], gpus: list[Gpu], hcas: list[Hca], link_layer: str) -> list[JsonDict]:
    """
    One row per NUMA node with its CPUs, GPUs, the HCAs on link_layer (NCCL's, cfg.nic.link_layer), and the other
    HCAs. Devices without NUMA affinity get a row of their own, numa None, last.
    """
    rows: dict[int | None, dict[str, Any]] = {}

    def row(numa: int | None) -> dict[str, Any]:
        empty = {"socket": None, "numa": numa, "cpus": None, "gpus": [], "hcas": [], "other": []}
        return rows.setdefault(numa, empty)

    def bus(device: Gpu | Hca) -> str:
        return device.address.short if device.address else "?"

    for node in nodes:
        row(node.numa_node).update(socket=node.socket, cpus=node.cpus)
    for device in gpus:
        row(device.numa_node)["gpus"].append(f"GPU{device.index} ({bus(device)})")
    for hca in hcas:
        if link_layer not in hca.link_layers:
            layers = ", ".join(hca.link_layers) or "unknown link layer"
            row(hca.numa_node)["other"].append(f"{hca.name} ({bus(hca)}, {layers})")
            continue
        shared = [f"GPU{device.index}" for device in gpus if hca.same_root_port(device)]
        switch = f" same PCIe switch as {', '.join(shared)}" if shared else ""
        row(hca.numa_node)["hcas"].append(f"{hca.name} ({bus(hca)}){switch}")
    return [
        {key: "; ".join(value) if isinstance(value, list) else value for key, value in rows[numa].items()}
        for numa in sorted(rows, key=lambda numa: (numa is None, numa or 0))
    ]
