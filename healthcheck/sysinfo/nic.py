"""NIC configuration: InfiniBand HCAs and their ports, network interfaces, GPUDirect RDMA."""

import re
from pathlib import Path

from healthcheck.util.file import JsonDict, read_sys


def _rate_gbps(rate: str | None) -> float | None:
    """ "400 Gb/sec (4X NDR)" -> 400.0"""
    match = re.match(r"\s*([\d.]+)", rate or "")
    return float(match.group(1)) if match else None


def collect() -> JsonDict:
    ib: list[JsonDict] = []
    for dev in sorted(Path("/sys/class/infiniband").glob("*")):
        ports: list[JsonDict] = []
        for port in sorted((dev / "ports").glob("*"), key=lambda path: int(path.name)):
            rate = read_sys(port / "rate")
            ports.append(
                {
                    "port": int(port.name),
                    "state": (read_sys(port / "state") or "").split(":")[-1].strip(),
                    "phys_state": (read_sys(port / "phys_state") or "").split(":")[-1].strip(),
                    "rate": rate,
                    "rate_gbps": _rate_gbps(rate),
                    "link_layer": read_sys(port / "link_layer"),
                }
            )
        ib.append(
            {
                "name": dev.name,
                "hca_type": read_sys(dev / "hca_type"),
                "fw_ver": read_sys(dev / "fw_ver"),
                "board_id": read_sys(dev / "board_id"),
                "netdevs": sorted(netdev.name for netdev in (dev / "device" / "net").glob("*")),
                "ports": ports,
            }
        )

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
    return {"ib": ib, "net": net, "nvidia_peermem": Path("/sys/module/nvidia_peermem").exists()}
