"""Step 1: node configuration (CPU, RAM, GPU, NIC, software) for threshold and homogeneity checks."""

import json
import platform
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import Any

import pynvml

from healthcheck.check import Availability, Finding, HealthCheckBase, Report, Severity, per_host
from healthcheck.config import GpuLimits, Homogeneity, Nic
from healthcheck.sysinfo import bios, cpu_power, gpu, job, nic, software, topology
from healthcheck.util.file import JsonDict, read_sys
from healthcheck.util.hostlist import ByHost
from healthcheck.util.nvml import session
from healthcheck.util.status import Status

# The key of the NUMA layout table in collect()'s result; node.local() moves it to the node's own numa/<host>.json
NUMA_LAYOUT = "numa_layout"


class HealthCheck(HealthCheckBase):
    def __init__(self, host: str, link_layer: str) -> None:
        self.host = host
        self.link_layer = link_layer

    @property
    def enabled(self) -> bool:
        return True

    def available(self) -> Availability:
        """NVML has to initialize: the GPU fields come from it."""
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as error:
            return Availability(False, str(error))
        pynvml.nvmlShutdown()
        return Availability(True)

    def collect(self) -> JsonDict:
        """The link layer is the one NCCL's NICs use: the NUMA layout lists those HCAs apart from the rest."""
        total = topology.total_bytes()
        placements = topology.placements()
        nodes = topology.numa_nodes(placements)
        with session():  # one NVML init for devices(), info() and counters(): slow without persistence mode
            gpus = gpu.devices()
            gpu_info = gpu.info()
            counters = gpu.counters()
        hcas = nic.hcas()
        return {
            "host": self.host,
            "kernel": platform.release(),
            "os_release": software.os_release(),
            "cmdline": read_sys("/proc/cmdline"),
            "bios": bios.collect(),
            "cpu": [socket.to_json() for socket in topology.sockets(placements)],
            "cpu_power": cpu_power.collect(),
            "ram_gib": round(total / 2**30, 1) if total else None,
            "allocation": job.allocation()._asdict(),
            "numa_nodes": [node.to_json() for node in nodes],
            "hugepages": topology.meminfo("HugePages_Total"),
            "gpu": {"count": len(gpus), **gpu_info, "devices": [device.to_json() for device in gpus]},
            "counters": [snapshot.to_json() for snapshot in counters],
            "nic": {"ib": [hca.to_json() for hca in hcas], **nic.collect()},
            "software": software.collect(),
            NUMA_LAYOUT: topology.numa_layout(nodes, gpus, hcas, self.link_layer),
        }

    @staticmethod
    def report(results: ByHost[JsonDict], gpu: GpuLimits, nic: Nic, homogeneity: Homogeneity, gpus: int) -> Report:
        """Sysinfo reads the GPU and NIC thresholds, the expected GPU count, and the homogeneity key paths."""
        usable = {
            host: data
            for host, data in results.items()
            if data.get("status") != Status.SKIPPED
        }
        findings = per_host(usable, gpu, "sysinfo", lambda data, _gpu: _findings(data, gpu, nic, gpus))
        for host, extra in _homogeneity(usable, homogeneity).items():
            findings.setdefault(host, []).extend(extra)
        return Report(findings, {})


def resolve(data: Any, path: str) -> Any:
    """
    Values at a dotted key path; "[*]" spans a list or dict, giving a tuple.

    >>> resolve({"nic": {"ib": [{"fw": "1"}, {"fw": "2"}]}}, "nic.ib[*].fw")
    ('1', '2')
    """
    head, _, rest = path.partition(".")
    spread = head.endswith("[*]")
    key = head.removesuffix("[*]")
    if not isinstance(data, dict) or key not in data:
        return None
    value = data[key]
    if spread:
        items = value.values() if isinstance(value, dict) else value or []
        return tuple(resolve(item, rest) if rest else item for item in items)
    return resolve(value, rest) if rest else value


def _findings(data: JsonDict, gpu_cfg: GpuLimits, nic_cfg: Nic, gpus: int) -> Iterator[Finding]:
    devices = data["gpu"]
    if devices["count"] != gpus:
        yield Finding("fail", "gpu.count", f"{devices['count']} GPUs, expected {gpus}")
    for device in devices["devices"]:
        index = device["index"]
        if gpu_cfg.require_ecc and device["ecc_enabled"] is False:
            yield Finding("fail", "gpu.ecc", f"GPU{index} ECC disabled")
        if device["mig_enabled"]:
            yield Finding("fail", "gpu.mig", f"GPU{index} MIG enabled")
        if (device["pcie_gen_max"] or 0) < gpu_cfg.min_pcie_gen:
            gen = device["pcie_gen_max"]
            yield Finding("fail", "gpu.pcie", f"GPU{index} max PCIe gen {gen} < {gpu_cfg.min_pcie_gen}")
        if (device["pcie_width"] or 0) < gpu_cfg.min_pcie_width:
            yield Finding("fail", "gpu.pcie", f"GPU{index} PCIe x{device['pcie_width']} < x{gpu_cfg.min_pcie_width}")
        if device["nvlinks_active"] < gpu_cfg.min_nvlinks:
            active_links = device["nvlinks_active"]
            yield Finding("fail", "gpu.nvlink", f"GPU{index} {active_links} NVLinks < {gpu_cfg.min_nvlinks}")
    for counters in data.get("counters", []):
        index = counters["index"]
        if (counters["ecc_volatile"]["uncorrected"] or 0) > gpu_cfg.max_volatile_uncorrected_ecc:
            yield Finding("fail", "gpu.ecc", f"GPU{index} {counters['ecc_volatile']['uncorrected']} uncorrected ECC")
        if rows := counters["remapped_rows"]:
            if rows["failure"] > gpu_cfg.max_remap_failures:
                yield Finding("fail", "gpu.remap", f"GPU{index} row remap failure")
            if rows["pending"] > gpu_cfg.max_remap_pending:
                yield Finding("fail", "gpu.remap", f"GPU{index} row remap pending (needs reset)")

    active = [
        (dev["name"], port)
        for dev in data["nic"]["ib"]
        for port in dev["ports"]
        if port["state"] == "ACTIVE"
        and port["link_layer"] == nic_cfg.link_layer
    ]
    if len(active) < nic_cfg.min_active_ib_ports:
        yield Finding(
            "fail", "nic.ports", f"{len(active)} active {nic_cfg.link_layer} ports < {nic_cfg.min_active_ib_ports}"
        )
    for nic_name, port in active:
        if (port["rate_gbps"] or 0) < nic_cfg.min_rate_Gbps:
            rate = f"{nic_name}/{port['port']} {port['rate']} < {nic_cfg.min_rate_Gbps} Gb/s"
            yield Finding("fail", "nic.rate", rate)


def _homogeneity(sysinfos: ByHost[JsonDict], cfg: Homogeneity) -> ByHost[list[Finding]]:
    """fail and warn key paths into sysinfo should match on every node.

    Nodes outside the majority value are flagged; with no value on more than half the nodes, all of them are.
    """
    out: ByHost[list[Finding]] = defaultdict(list)
    severity: Severity
    for severity, keys in (("fail", cfg.fail), ("warn", cfg.warn)):
        for key in keys:
            values = {
                host: json.dumps(resolve(data, key), default=str)
                for host, data in sysinfos.items()
            }
            if len(set(values.values())) < 2:
                continue
            most_common, count = Counter(values.values()).most_common(1)[0]
            if count * 2 <= len(values):
                for host, value in values.items():
                    out[host].append(Finding(severity, "homogeneity", f"{key}={value} (no majority)"))
                continue
            for host, value in values.items():
                if value != most_common:
                    out[host].append(Finding(severity, "homogeneity", f"{key}={value} (majority {most_common})"))
    return out
