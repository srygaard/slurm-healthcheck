"""Step 2c: busy loops on a share of the job's CPUs while the dcgm step loads the GPUs, watching each CPU package.

The CPUs stay busy until the dcgm step completes. A package is a socket: its throttle counters and its busy cores'
clock are read apart from the other packages', so a cap on one socket is not averaged away.
"""

import contextlib
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pynvml

from healthcheck.check import Availability, Finding, HealthCheckBase, Report, median
from healthcheck.config import CpuLoad
from healthcheck.sysinfo.gpu import watts
from healthcheck.sysinfo.topology import cpulist, placements
from healthcheck.util.file import JsonDict, read_sys
from healthcheck.util.hostlist import ByHost
from healthcheck.util.nvml import session
from healthcheck.util.status import Status

def cpu_mhz(cpuinfo: Path = Path("/proc/cpuinfo")) -> dict[int, float]:
    """Each CPU's current clock, as /proc/cpuinfo's "processor" and "cpu MHz" lines give it."""
    out = {}
    processor = None
    for line in (read_sys(cpuinfo) or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "processor":
            processor = int(value)
        elif key.strip() == "cpu MHz" and processor is not None:
            out[processor] = float(value)
    return out


@dataclass
class CpuPackage:
    """The job's CPUs on one socket: which of them burn, their clock, and the socket's throttle counters."""

    socket: int
    cpus: list[int]
    busy: list[int]
    sys_root: Path = Path("/sys")
    burners: list[subprocess.Popen] = field(default_factory=list)

    @staticmethod
    def spread(cpus: list[int], frac: float) -> list[int]:
        """
        frac of cpus, evenly spaced so the package keeps some idle CPUs.

        >>> CpuPackage.spread(list(range(8)), 0.75)
        [0, 1, 2, 4, 5, 6]
        """
        count = round(len(cpus) * frac)
        if not count:
            return []
        return sorted({cpus[index * len(cpus) // count] for index in range(count)})

    @staticmethod
    def start_burner(cpu: int) -> subprocess.Popen:
        """A busy loop pinned to cpu at SCHED_IDLE, so any other thread that wants the CPU gets it at once."""
        # ~78 us of C multiply loops each, on 4 KB numbers that stay in L1/L2. Heavier than an empty loop, which is
        # mostly interpreter dispatch, but scalar, unlike AVX math that would hold every node at its RAPL limit.
        burn = "base = 3\nwhile True:\n    base**20000"
        process = subprocess.Popen([sys.executable, "-c", burn])
        try:
            os.sched_setaffinity(process.pid, {cpu})
            os.sched_setscheduler(process.pid, os.SCHED_IDLE, os.sched_param(0))
        except OSError:
            process.kill()
            process.wait()
            raise
        return process

    @classmethod
    def of_job(cls, cpus: list[int], frac: float, sys_root: Path = Path("/sys")) -> list["CpuPackage"]:
        """One package per socket the job's cpus sit on, each with its own frac of them busy. A CPU with no topology
        (a container, a VM) counts under socket 0."""
        socket_of = {
            placement.cpu: placement.socket
            for placement
            in placements(sys_root)
        }
        by_socket: dict[int, list[int]] = {}
        for cpu in sorted(cpus):
            by_socket.setdefault(socket_of.get(cpu, 0), []).append(cpu)
        return [
            cls(socket, mine, cls.spread(mine, frac), sys_root)
            for socket, mine
            in sorted(by_socket.items())
        ]

    def burn_start(self) -> None:
        for cpu in self.busy:
            self.burners.append(self.start_burner(cpu))

    def burn_stop(self) -> None:
        for burner in self.burners:
            burner.kill()
        for burner in self.burners:
            burner.wait()

    def busy_mhz(self, clocks: dict[int, float]) -> float | None:
        """The busy cores' mean clock, from one cpu_mhz() read."""
        values = [clocks[cpu] for cpu in self.busy if cpu in clocks]
        return round(statistics.mean(values)) if values else None

    @staticmethod
    def median_mhz(values: list[float]) -> int | None:
        """A package's clock over a window. The median keeps a few halted readings from setting it."""
        return round(statistics.median(values)) if values else None

    def throttles(self) -> dict[str, int] | None:
        """
        Throttle events since boot, by cause: the core counters summed over the package's CPUs, plus its package
        counter, read once as the highest over those CPUs. None when no CPU has a thermal_throttle directory (a VM, or
        a kernel that does not expose it), so a missing counter is not a zero.
        """
        cpu_root = self.sys_root / "devices/system/cpu"
        dirs = [
            path
            for cpu
            in self.cpus
            if (path := cpu_root / f"cpu{cpu}/thermal_throttle").is_dir()
        ]
        if not dirs:
            return None
        # thermal_throttle files in each CPU's sysfs directory, by cause. core_* belong to the core; package_* are the
        # same value in every CPU's directory.
        core = {"thermal": "core_throttle_count", "power_limit": "core_power_limit_count"}
        package = {"thermal": "package_throttle_count", "power_limit": "package_power_limit_count"}
        return {
            cause: sum(self._count(path / core[cause]) for path in dirs)
            + max(self._count(path / package[cause]) for path in dirs)
            for cause in core
        }

    @staticmethod
    def _count(path: Path) -> int:
        value = read_sys(path)
        return int(value) if value else 0

    @staticmethod
    def delta(before: dict[str, int] | None, after: dict[str, int] | None) -> dict[str, int] | None:
        """
        The events between two throttles() reads; None when either found no counters.

        >>> CpuPackage.delta({"thermal": 1, "power_limit": 4}, {"thermal": 1, "power_limit": 7})
        {'thermal': 0, 'power_limit': 3}
        >>> CpuPackage.delta(None, {"thermal": 0, "power_limit": 0}) is None
        True
        """
        if None in (before, after):
            return None
        return {
            cause: after[cause] - before[cause]
            for cause
            in after
        }

    def throttle_delta(self, before: dict[str, int] | None) -> dict[str, int] | None:
        """The events since a throttles() read taken before the burn."""
        return self.delta(before, self.throttles())

    def record(
        self,
        before: dict[str, int] | None,
        full_power_mhz: float | None,
        baseline_mhz: float | None
    ) -> JsonDict:
        """This package's part of the node JSON."""
        return {
            "busy_cpus": cpulist(self.busy),
            "throttle_delta": self.throttle_delta(before),
            "full_power_mhz": full_power_mhz,
            "baseline_mhz": baseline_mhz,
        }

    @staticmethod
    def label(socket: str) -> str:
        """The prefix a finding uses. "all" is a record from before packages, which had one clock for the node."""
        return "" if socket == "all" else f"package {socket}: "

    @classmethod
    def judge(cls, socket: str, package: JsonDict, floor: float, cfg: CpuLoad) -> Iterator[Finding]:
        """This package's findings: a throttle event or a clock under the run's floor warns. A clock under a share of
        the package's own clock before the GPUs loaded up warns below cfg.warn_freq_frac_of_baseline and fails below
        cfg.fail_freq_frac_of_baseline."""
        counts = package.get("throttle_delta")
        if counts and any(counts.values()):
            events = f"{counts['thermal']} thermal and {counts['power_limit']} power-limit throttle events"
            yield Finding("warn", "cpu_load.throttle", f"{cls.label(socket)}{events}")
        mhz = package.get("full_power_mhz")
        if mhz is not None and mhz < floor:
            detail = f"{cls.label(socket)}busy-core clock {mhz} < {floor:.0f} MHz"
            yield Finding("warn", "cpu_load.median", detail)
        baseline = package.get("baseline_mhz")
        if mhz is None or not baseline:
            return
        for severity, frac in (("fail", cfg.fail_freq_frac_of_baseline), ("warn", cfg.warn_freq_frac_of_baseline)):
            if mhz < frac * baseline:
                clock = f"busy-core clock {mhz} MHz at full GPU power < {frac:.0%} of {baseline} MHz"
                detail = f"{cls.label(socket)}{clock}"
                yield Finding(severity, "cpu_load.sag", detail)
                break


def _clocks(sample: JsonDict) -> dict[str, float | None]:
    """A sample's clock by package. A run from before packages stored one busy_mhz, which counts as package "all"."""
    return sample["mhz"] if "mhz" in sample else {"all": sample.get("busy_mhz")}


def summarize(samples: list[JsonDict], cfg: CpuLoad) -> JsonDict:
    """
    Each package's median busy-core clock at full GPU power: the samples drawing at least full_power_frac of the run's
    peak GPU power, if they span min_full_power_s. Its baseline is the median over the samples under
    baseline_power_frac of that peak, the ramp before and after; none when there are fewer than two. The median keeps
    a few halted 200 MHz readings from setting either clock. The first sample is left out: it is taken as the burners
    start, so it can still read idle clocks.

    >>> rows = ((0, 3900, 300), (5, 3200, 1000), (10, 3200, 2000), (15, 2400, 7000), (20, 200, 7000), (25, 2400, 7000))
    >>> samples = [{"t_s": t, "mhz": {"0": mhz}, "gpu_w": w} for t, mhz, w in rows]
    >>> summarize(samples, CpuLoad(min_full_power_s=15))
    {'full_power_samples': 3, 'full_power_mhz': {'0': 2400}, 'baseline_mhz': {'0': 3200}}
    >>> summarize(samples, CpuLoad(min_full_power_s=25))
    {'full_power_samples': 3, 'full_power_mhz': {}, 'baseline_mhz': {}}
    """
    samples = samples[1:]  # the first reading is taken as the burners start, before the cores settle to the burn clock
    peak = max((sample["gpu_w"] or 0 for sample in samples), default=0)
    loaded = [sample for sample in samples if peak and (sample["gpu_w"] or 0) >= cfg.full_power_frac * peak]
    if len(loaded) * cfg.interval_s < cfg.min_full_power_s:
        return {"full_power_samples": len(loaded), "full_power_mhz": {}, "baseline_mhz": {}}
    # A sample whose GPU power couldn't be read is in neither group: it may have been taken at full power
    unloaded = [
        sample
        for sample in samples
        if sample["gpu_w"] is not None and sample["gpu_w"] < cfg.baseline_power_frac * peak
    ]
    return {
        "full_power_samples": len(loaded),
        "full_power_mhz": _median_clocks(loaded),
        "baseline_mhz": _median_clocks(unloaded) if len(unloaded) >= 2 else {},
    }


def _median_clocks(samples: list[JsonDict]) -> dict[str, int | None]:
    """Each package's median clock over these samples."""
    clocks = [_clocks(sample) for sample in samples]
    sockets = sorted({socket for sample in clocks for socket in sample})
    return {
        socket: CpuPackage.median_mhz([sample[socket] for sample in clocks if sample.get(socket) is not None])
        for socket in sockets
    }


class HealthCheck(HealthCheckBase):
    def __init__(self, cfg: CpuLoad, stop_event: threading.Event, cpus: list[int]) -> None:
        self.cfg = cfg
        self.stop_event = stop_event
        self.packages = CpuPackage.of_job(cpus, cfg.busy_frac)

    @classmethod
    def missing(cls, _name: str) -> None:
        """A node JSON from before this step existed has no cpu_load key, and that is not a fault."""
        return

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def available(self) -> Availability:
        return Availability(True)

    def collect(self) -> JsonDict:
        """Hold the load until stop_event is set; cpus are the job's, os.sched_getaffinity(0).

        The burners run at SCHED_IDLE, so the other phase A steps, DCGM and the Xid watcher never wait for a CPU.
        They stop when dcgm returns and, if it failed, its test lets go of the GPUs; both are bounded by timeouts.
        """
        before = {
            package.socket: package.throttles()
            for package
            in self.packages
        }
        samples = []
        start = time.monotonic()
        try:
            for package in self.packages:
                package.burn_start()
            with contextlib.ExitStack() as nvml:
                with contextlib.suppress(pynvml.NVMLError):  # gpu_w is None without a driver
                    nvml.enter_context(session())
                while not self.stop_event.is_set():
                    clocks = cpu_mhz()
                    samples.append(
                        {
                            "t_s": round(time.monotonic() - start, 1),
                            "gpu_w": watts(),
                            "mhz": {
                                str(package.socket): package.busy_mhz(clocks)
                                for package in self.packages
                            },
                        }
                    )
                    self.stop_event.wait(self.cfg.interval_s)
        finally:
            for package in self.packages:
                package.burn_stop()
        held_s = round(time.monotonic() - start, 1)
        # A short hold (dcgm skipped, failed early, or ran without targeted_power) still counts throttle events;
        # summarize leaves each package's clock None when full GPU power didn't span min_full_power_s.
        summary = summarize(samples, self.cfg)
        return {
            "status": Status.OK,
            "held_s": held_s,
            "full_power_samples": summary["full_power_samples"],
            "packages": {
                str(package.socket): package.record(
                    before[package.socket],
                    summary["full_power_mhz"].get(str(package.socket)),
                    summary["baseline_mhz"].get(str(package.socket)),
                )
                for package in self.packages
            },
            "samples": samples,
        }

    @staticmethod
    def report(results: ByHost[JsonDict], cfg: CpuLoad, _name: str) -> Report:
        graded = {
            host: {**data, "packages": _packages(data, cfg)}
            for host, data in results.items()
            if data.get("status") != Status.SKIPPED
        }
        clocks = [
            package["full_power_mhz"]
            for data in graded.values()
            for package in data["packages"].values()
            if package.get("full_power_mhz")
        ]
        mid = median(clocks)
        floor = (mid or 0) * cfg.min_frac_of_median
        findings: ByHost[list[Finding]] = {}
        for host, data in graded.items():
            found = [
                finding
                for socket, package in data["packages"].items()
                for finding in CpuPackage.judge(socket, package, floor, cfg)
            ]
            if found:
                findings[host] = found
        return Report(findings, {"cpu_load_busy_mhz": mid} if clocks else {})


def _packages(data: JsonDict, cfg: CpuLoad) -> dict[str, JsonDict]:
    """
    A node's packages, with each full-power clock recomputed from its samples, so a report applies the current
    thresholds to a run recorded under older ones. A record from before packages is one package "all", with its
    integer throttle_delta counted as thermal events. A record without samples keeps the clock it was written with.

    >>> _packages({"full_power_mhz": 2300, "throttle_delta": 2}, CpuLoad())
    {'all': {'full_power_mhz': 2300, 'throttle_delta': {'thermal': 2, 'power_limit': 0}}}
    """
    packages = data.get("packages")
    if packages is None:
        legacy = data.get("throttle_delta")
        throttles = {"thermal": legacy, "power_limit": 0} if legacy is not None else None
        packages = {"all": {"full_power_mhz": data.get("full_power_mhz"), "throttle_delta": throttles}}
    if data.get("samples"):
        summary = summarize(data["samples"], cfg)
        packages = {
            socket: {
                **package,
                "full_power_mhz": summary["full_power_mhz"].get(socket),
                "baseline_mhz": summary["baseline_mhz"].get(socket),
            }
            for socket, package in packages.items()
        }
    return packages
