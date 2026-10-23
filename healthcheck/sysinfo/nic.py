"""NIC configuration: InfiniBand HCAs, their ports and PCI locality, network interfaces, GPUDirect RDMA."""

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from healthcheck.util.device import BusId, PciDevice
from healthcheck.util.file import JsonDict, read_sys


class LinkLayer(enum.StrEnum):
    INFINIBAND = "InfiniBand"
    ETHERNET = "Ethernet"


@dataclass(frozen=True, kw_only=True)
class Hca(PciDevice):
    """An RDMA device, e.g. mlx5_2; an Ethernet port of a ConnectX card is one too."""

    hca_type: str | None
    fw_ver: str | None
    board_id: str | None
    netdevs: list[str]
    ports: list[JsonDict]

    @property
    def link_layers(self) -> list[LinkLayer]:
        """Its ports' known link layers; mlx5 gives each port its own device, but a VPI card's ports can differ."""
        return [LinkLayer(port["link_layer"]) for port in self.ports if port["link_layer"] in LinkLayer]

    @classmethod
    def from_infiniband(cls, dev: Path, root: Path = Path("/sys")) -> Self:
        """From its /sys/class/infiniband/<name> directory."""
        return cls.from_sysfs(
            BusId.parse((dev / "device").resolve().name),
            root,
            name=dev.name,
            hca_type=read_sys(dev / "hca_type"),
            fw_ver=read_sys(dev / "fw_ver"),
            board_id=read_sys(dev / "board_id"),
            netdevs=sorted(netdev.name for netdev in (dev / "device" / "net").glob("*")),
            ports=[_port(port) for port in sorted((dev / "ports").glob("*"), key=lambda path: int(path.name))],
        )


def _rate_gbps(rate: str | None) -> float | None:
    """ "400 Gb/sec (4X NDR)" -> 400.0"""
    match = re.match(r"\s*([\d.]+)", rate or "")
    return float(match.group(1)) if match else None


def _port(port: Path) -> JsonDict:
    rate = read_sys(port / "rate")
    return {
        "port": int(port.name),
        "state": (read_sys(port / "state") or "").split(":")[-1].strip(),
        "phys_state": (read_sys(port / "phys_state") or "").split(":")[-1].strip(),
        "rate": rate,
        "rate_gbps": _rate_gbps(rate),
        "link_layer": read_sys(port / "link_layer"),
    }


def hcas(root: Path = Path("/sys")) -> list[Hca]:
    return [
        Hca.from_infiniband(dev, root)
        for dev
        in sorted((root / "class" / "infiniband").glob("*"))
    ]


def collect() -> JsonDict:
    """Network interfaces and GPUDirect RDMA; sysinfo adds the HCAs."""
    net = {}
    for iface in sorted(Path("/sys/class/net").glob("*")):
        if iface.name == "lo":
            continue
        speed = read_sys(iface / "speed")
        net[iface.name] = {
            "mtu": int(mtu) if (mtu := read_sys(iface / "mtu")) else None,
            "speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit() else None,
            "operstate": read_sys(iface / "operstate"),
            "ipoib_mode": read_sys(iface / "mode"),
        }
    return {
        "net": net,
        "nvidia_peermem": Path("/sys/module/nvidia_peermem").exists()
    }
