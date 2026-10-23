"""Node hardware as typed records: a device's NUMA node, and for PCI devices where they sit on the PCIe tree."""

import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self

from healthcheck.util.file import JsonDict, read_sys


@dataclass(frozen=True, order=True)
class BusId:
    """A PCI address, domain:bus:device.function."""

    domain: int
    bus: int
    device: int
    function: int

    @classmethod
    def parse(cls, text: str | None) -> Self | None:
        """
        NVML's or sysfs's form; None if text isn't a PCI address.

        >>> BusId.parse("00000000:4C:00.0") == BusId.parse("0000:4c:00.0")
        True
        >>> BusId.parse("pci0000:00") is None
        True
        """
        match = re.fullmatch(r"([0-9a-f]+):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])", (text or "").lower())
        return cls(*(int(group, 16) for group in match.groups())) if match else None

    @property
    def short(self) -> str:
        """
        >>> BusId.parse("00000000:4C:00.0").short
        '4c:00.0'
        """
        return f"{self.bus:02x}:{self.device:02x}.{self.function:x}"

    def __str__(self) -> str:
        """The sysfs form, "0000:4c:00.0"."""
        return f"{self.domain:04x}:{self.short}"


@dataclass(frozen=True, kw_only=True)
class Device:
    """A piece of node hardware and the NUMA node it's local to."""

    name: str | None
    numa_node: int | None  # None if not local to one NUMA node, or unknown

    def to_json(self) -> JsonDict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, kw_only=True)
class PciDevice(Device):
    address: BusId | None
    vendor: str | None  # "0x10de"
    device: str | None  # "0x2330"
    root_port: BusId | None  # None if the device sits on the host bridge itself

    @classmethod
    def from_sysfs(cls, address: BusId | None, root: Path = Path("/sys"), **values: Any) -> Self:
        """The PCI fields from sysfs, the subclass's from values; an unknown or missing device leaves them None."""
        if address is None:
            return cls(address=None, vendor=None, device=None, numa_node=None, root_port=None, **values)
        path = root / "bus" / "pci" / "devices" / str(address)
        numa = read_sys(path / "numa_node")
        # /sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/.../0000:03:00.0: host bridge, root port, ..., device
        parts = path.resolve().parts
        host_bridge = next(
            (index for index, part in enumerate(parts) if re.fullmatch(r"pci[0-9a-f]+:[0-9a-f]+", part)), None
        )
        below = parts[host_bridge + 1 :] if host_bridge is not None else ()
        return cls(
            address=address,
            vendor=read_sys(path / "vendor"),
            device=read_sys(path / "device"),
            numa_node=int(numa) if numa and numa.isdigit() else None,  # the kernel writes -1 for no affinity
            root_port=BusId.parse(below[0]) if len(below) > 1 else None,
            **values,
        )

    def same_root_port(self, other: "PciDevice") -> bool:
        """Both below one PCIe root port, so on one PCIe switch (`nvidia-smi topo -m` PIX or PXB)."""
        return self.root_port is not None and self.root_port == other.root_port

    def to_json(self) -> JsonDict:
        """The subclass's fields, with the PCI ones and numa_node under "pci"."""
        out = dict(super().to_json())
        pci = {key: out.pop(key) for key in ("address", "vendor", "device", "numa_node", "root_port")}
        return {**out, "pci": {key: str(value) if isinstance(value, BusId) else value for key, value in pci.items()}}
