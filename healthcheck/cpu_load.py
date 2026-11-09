"""Step 2c: busy loops on a share of the job's CPUs until dcgm returns, watching each CPU package (socket) apart.

CpuPackage drives the hardware. Summary splits the run's samples by GPU power and takes each package's clocks from them.
PackageRecord is one package's part of the node JSON: collect() writes it and report() reads it back.
"""

import contextlib
import itertools
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Self, TypedDict

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
    """
    The job's CPUs on one socket, live: which of them burn, their clock, and the socket's thermal throttling. Each
    socket is read on its own, so a cap on one is not averaged away by the others.
    """

    socket: int
    cpus: list[int]
    busy: list[int]
    burn_throttles: int | None = field(default=None, init=False)  # episodes over the last burning() block

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
    def of_job(cls, cpus: list[int], frac: float, sys_root: Path = Path("/sys")) -> list[Self]:
        """One package per socket the job's cpus sit on, each with frac of its CPUs busy."""
        socket_of = {placement.cpu: placement.socket for placement in placements(sys_root)}
        by_socket: dict[int, list[int]] = {}
        for cpu in sorted(cpus):
            # A CPU with no topology (a container, a VM) counts under socket 0
            by_socket.setdefault(socket_of.get(cpu, 0), []).append(cpu)
        return [cls(socket, mine, cls.spread(mine, frac)) for socket, mine in sorted(by_socket.items())]

    @contextlib.contextmanager
    def burning(self) -> Iterator[None]:
        """A burner on each busy CPU for the length of the block, counting the package's thermal throttling over it."""
        self.burn_throttles = None
        before = self.thermal_throttles()
        burners: list[subprocess.Popen] = []
        try:
            burners.extend(self.start_burner(cpu) for cpu in self.busy)  # keeps those started before a failure
            yield
        finally:
            for burner in burners:
                burner.kill()
            for burner in burners:
                burner.wait()
            after = self.thermal_throttles()
            # None when the package has no counter: a missing counter is not a zero
            if before is not None and after is not None:
                self.burn_throttles = after - before

    def busy_mhz(self, clocks: dict[int, float]) -> float | None:
        """The busy cores' mean clock, from one cpu_mhz() read."""
        values = [clocks[cpu] for cpu in self.busy if cpu in clocks]
        return round(statistics.mean(values)) if values else None

    def thermal_throttles(self, cpu_sysfs: Path = Path("/sys/devices/system/cpu")) -> int | None:
        """
        The package's thermal throttling episodes since boot, each entry into throttling counted once; None without the
        counter (a VM, or a kernel without thermal_throttle).
        """
        # Every CPU's thermal_throttle directory shows the same package count, so the first CPU's stands for them all.
        # The per-core and power-limit counters beside it are not read: the kernel only creates the power-limit ones
        # when booted with int_pln_enable, which Trillium's B200 nodes are not.
        count = read_sys(cpu_sysfs / f"cpu{self.cpus[0]}/thermal_throttle/package_throttle_count")
        return int(count) if count else None


class BurnSample(TypedDict):
    """One reading during the burn, as the node JSON's "samples" holds it."""

    t_s: float  # since the burners started
    gpu_w: float | None  # all GPUs together; None when one couldn't be read
    mhz: dict[str, float | None]  # each package's busy-core clock, by socket


class Clocks(NamedTuple):
    """A package's median busy-core clock at full GPU power and at baseline."""

    full_power_mhz: int | None
    baseline_mhz: int | None


@dataclass(frozen=True)
class Summary:
    """A run's samples, split by GPU power against the run's peak; of() takes each package's clocks from them."""

    samples: list[BurnSample]
    full_power_frac: float  # of the peak: samples at or above it are at full GPU power
    baseline_power_frac: float  # of the peak: samples under it are the baseline
    min_full_power_s: float  # less time than this at full power gives no clocks

    @classmethod
    def from_samples(cls, samples: list[BurnSample], cfg: CpuLoad) -> Self:
        """
        All but the first sample, which is taken as the burners start, before the cores settle to the burn clock; the
        thresholds come from cfg.
        """
        return cls(samples[1:], cfg.full_power_frac, cfg.baseline_power_frac, cfg.min_full_power_s)

    @cached_property
    def peak(self) -> float:
        """The run's peak GPU power."""
        return max((sample["gpu_w"] or 0 for sample in self.samples), default=0)

    @cached_property
    def loaded(self) -> list[BurnSample]:
        """The samples at full GPU power: full_power_frac of the peak."""
        return [s for s in self.samples if self.peak and (s["gpu_w"] or 0) >= self.full_power_frac * self.peak]

    @cached_property
    def unloaded(self) -> list[BurnSample]:
        """The baseline: the ramp before and after full power, under baseline_power_frac of the peak."""
        # A sample whose GPU power couldn't be read is left out: it may have been taken at full power
        limit = self.baseline_power_frac * self.peak
        return [sample for sample in self.samples if sample["gpu_w"] is not None and sample["gpu_w"] < limit]

    @property
    def full_power_samples(self) -> int:
        return len(self.loaded)

    @property
    def full_power_s(self) -> float:
        """Time at full GPU power, at the run's own sampling interval: a re-graded run may predate interval_s."""
        gaps = [later["t_s"] - earlier["t_s"] for earlier, later in itertools.pairwise(self.samples)]
        return self.full_power_samples * statistics.median(gaps) if gaps else 0.0

    def of(self, socket: str) -> Clocks:
        """
        A package's clocks: medians, so a few halted readings don't set them. Both None when full power didn't span
        min_full_power_s; the baseline None with under two samples.

        >>> rows = ((0, 3900, 300), (5, 3200, 1000), (10, 3200, 2000))  # start-up reading, then the baseline ramp
        >>> rows += ((15, 2400, 7000), (20, 200, 7000), (25, 2400, 7000))  # full power: 15 s
        >>> samples = [{"t_s": t, "mhz": {"0": mhz}, "gpu_w": w} for t, mhz, w in rows]
        >>> Summary.from_samples(samples, CpuLoad(min_full_power_s=15)).of("0")
        Clocks(full_power_mhz=2400, baseline_mhz=3200)
        >>> Summary.from_samples(samples, CpuLoad(min_full_power_s=25)).of("0")
        Clocks(full_power_mhz=None, baseline_mhz=None)
        """

        def median(samples: list[BurnSample]) -> int | None:
            values = [mhz for sample in samples if (mhz := sample["mhz"].get(socket)) is not None]
            return round(statistics.median(values)) if values else None

        if self.full_power_s < self.min_full_power_s:
            return Clocks(None, None)
        return Clocks(median(self.loaded), median(self.unloaded) if len(self.unloaded) >= 2 else None)


@dataclass(frozen=True)
class PackageRecord:
    """One package's part of a node JSON: what collect() writes and report() grades."""

    socket: str
    thermal_throttles: int | None  # package thermal throttling episodes during the burn
    full_power_mhz: int | None  # the median busy-core clock at full GPU power
    baseline_mhz: int | None  # and while the GPUs drew under baseline_power_frac of their peak
    busy_cpus: str | None = None  # a cpulist

    def to_json(self) -> JsonDict:
        """This package's entry under the node JSON's "packages", keyed there by socket."""
        return {
            "busy_cpus": self.busy_cpus,
            "thermal_throttles": self.thermal_throttles,
            "full_power_mhz": self.full_power_mhz,
            "baseline_mhz": self.baseline_mhz,
        }

    @classmethod
    def of_node(cls, data: JsonDict, cfg: CpuLoad) -> list[Self]:
        """
        A node's packages, with clocks recomputed from the samples under the current thresholds.

        >>> (record,) = PackageRecord.of_node({"packages": {"0": {"full_power_mhz": 2300}}}, CpuLoad())
        >>> record.socket, record.thermal_throttles, record.full_power_mhz
        ('0', None, 2300)
        """
        # Recomputing applies current thresholds to a run recorded under older ones; without samples, the record keeps
        # the clocks it was written with.
        summary = Summary.from_samples(data["samples"], cfg) if data.get("samples") else None
        records = []
        for socket, package in data.get("packages", {}).items():  # none in an error record
            # Older records hold {"thermal": n, ...} under throttle_delta, with per-core episodes in n
            old = package.get("throttle_delta") or {}
            throttles = package.get("thermal_throttles", old.get("thermal"))
            stored = Clocks(package.get("full_power_mhz"), package.get("baseline_mhz"))
            clocks = summary.of(socket) if summary else stored
            records.append(cls(socket, throttles, *clocks, package.get("busy_cpus")))
        return records

    @property
    def label(self) -> str:
        """The prefix a finding uses."""
        return f"package {self.socket}: "

    def findings(self, floor: float, fail_frac: float, warn_frac: float) -> Iterator[Finding]:
        """
        Thermal throttling, and a busy-core clock under the run's floor or sagging under fail_frac or warn_frac of its
        baseline.
        """
        if self.thermal_throttles:
            events = f"{self.thermal_throttles} thermal throttling episodes"
            yield Finding("warn", "cpu_load.throttle", f"{self.label}{events}")
        mhz = self.full_power_mhz
        baseline = self.baseline_mhz
        if mhz is not None and mhz < floor:
            yield Finding("warn", "cpu_load.median", f"{self.label}busy-core clock {mhz} < {floor:.0f} MHz")
        if mhz is None or not baseline:
            return
        # Sag: the median against the package's own clock before the GPUs loaded up
        for severity, frac in (("fail", fail_frac), ("warn", warn_frac)):
            if mhz < frac * baseline:
                clock = f"busy-core clock {mhz} MHz at full GPU power < {frac:.0%} of {baseline} MHz"
                yield Finding(severity, "cpu_load.sag", f"{self.label}{clock}")
                break


class HealthCheck(HealthCheckBase):
    def __init__(self, cfg: CpuLoad, stop_event: threading.Event, cpus: list[int]) -> None:
        self.cfg = cfg
        self.stop_event = stop_event
        self.packages = CpuPackage.of_job(cpus, cfg.busy_frac)  # cpus are the job's, os.sched_getaffinity(0)

    @classmethod
    def missing(cls, _name: str) -> None:
        """A node JSON from before this step existed has no cpu_load key, and that is not a fault."""

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def available(self) -> Availability:
        return Availability(True)

    def collect(self) -> JsonDict:
        """Burn until stop_event is set, sampling each package's busy-core clock and the GPUs' power."""
        samples: list[BurnSample] = []
        start = time.monotonic()
        with contextlib.ExitStack() as stack:
            # SCHED_IDLE burners: the other phase A steps, DCGM and the Xid watcher never wait for a CPU. stop_event is
            # set when dcgm returns and, if it failed, its test lets go of the GPUs; both are bounded by timeouts.
            for package in self.packages:
                stack.enter_context(package.burning())
            with contextlib.suppress(pynvml.NVMLError):  # gpu_w is None without a driver
                stack.enter_context(session())
            while not self.stop_event.is_set():
                clocks = cpu_mhz()
                mhz = {str(package.socket): package.busy_mhz(clocks) for package in self.packages}
                samples.append(BurnSample(t_s=round(time.monotonic() - start, 1), gpu_w=watts(), mhz=mhz))
                self.stop_event.wait(self.cfg.interval_s)
        held_s = round(time.monotonic() - start, 1)
        # A short hold (dcgm skipped, failed early, or ran without targeted_power) still counts throttle events;
        # Summary.of() leaves each package's clocks None when full GPU power didn't span min_full_power_s.
        summary = Summary.from_samples(samples, self.cfg)
        packages = {}
        for package in self.packages:
            socket = str(package.socket)
            record = PackageRecord(socket, package.burn_throttles, *summary.of(socket), cpulist(package.busy))
            packages[socket] = record.to_json()
        return {
            "status": Status.OK,
            "held_s": held_s,
            "full_power_samples": summary.full_power_samples,
            "packages": packages,
            "samples": samples,
        }

    @staticmethod
    def report(results: ByHost[JsonDict], cfg: CpuLoad, _name: str) -> Report:
        graded = {
            host: PackageRecord.of_node(data, cfg)
            for host, data in results.items()
            if data.get("status") != Status.SKIPPED
        }
        # Every package on every node, so the floor comes from the run's median
        packages = list(itertools.chain.from_iterable(graded.values()))
        clocks = [package.full_power_mhz for package in packages if package.full_power_mhz]
        mid = median(clocks)
        floor = (mid or 0) * cfg.min_frac_of_median
        fail_frac = cfg.fail_freq_frac_of_baseline
        warn_frac = cfg.warn_freq_frac_of_baseline
        findings: ByHost[list[Finding]] = {}
        for host, records in graded.items():
            if found := [finding for record in records for finding in record.findings(floor, fail_frac, warn_frac)]:
                findings[host] = found
        return Report(findings, {"cpu_load_busy_mhz": mid} if clocks else {})
