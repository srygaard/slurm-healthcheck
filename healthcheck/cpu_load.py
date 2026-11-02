"""Step 2c: busy loops on a share of the job's CPUs while the dcgm step loads the GPUs, sampling the CPU clock.

The CPUs stay busy until the dcgm step completes.
"""

import contextlib
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pynvml

from healthcheck.check import Availability, Finding, HealthCheckBase, Report, Severity, below_median, per_host
from healthcheck.config import CpuLoad
from healthcheck.sysinfo.gpu import watts
from healthcheck.sysinfo.topology import cpulist
from healthcheck.util.file import JsonDict, read_sys
from healthcheck.util.hostlist import ByHost
from healthcheck.util.nvml import session
from healthcheck.util.status import Status


def spread(cpus: list[int], frac: float) -> list[int]:
    """
    frac of cpus, evenly spaced so every socket keeps some idle CPUs.

    >>> spread(list(range(8)), 0.75)
    [0, 1, 2, 4, 5, 6]
    """
    count = round(len(cpus) * frac)
    picked = {cpus[index * len(cpus) // count] for index in range(count)} if count else set()
    return sorted(picked)


# Big-integer powers: ~78 us of C multiply loops each, on 4 KB numbers that stay in L1/L2. Heavier than an empty
# loop, which is mostly interpreter dispatch, but scalar, unlike AVX math that would hold every node at its RAPL limit.
BURN = "base = 3\nwhile True:\n    base**20000"


def start_burner(cpu: int) -> subprocess.Popen:
    """A busy loop pinned to cpu at SCHED_IDLE, so any other thread that wants the CPU gets it at once."""
    process = subprocess.Popen([sys.executable, "-c", BURN])
    try:
        os.sched_setaffinity(process.pid, {cpu})
        os.sched_setscheduler(process.pid, os.SCHED_IDLE, os.sched_param(0))
    except OSError:
        process.kill()
        process.wait()
        raise
    return process


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


def throttle_count(root: Path = Path("/sys/devices/system/cpu")) -> int:
    """Thermal throttle events since boot, summed over every core and package counter."""
    paths = [*root.glob("cpu[0-9]*/thermal_throttle/core_throttle_count")]
    paths += root.glob("cpu[0-9]*/thermal_throttle/package_throttle_count")
    return sum(int(value) for path in paths if (value := read_sys(path)))


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values)) if values else None


def summarize(samples: list[JsonDict], cfg: CpuLoad) -> JsonDict:
    """
    The busy CPUs' highest clock, and at full GPU power (the samples drawing at least full_power_frac of the run's peak
    GPU power, if they span min_full_power_s) their mean clock and the share of readings more than low_frac below
    the highest. rg21701 swung between normal and low clocks, which a share tells apart from a healthy node's one-off
    dips better than a mean. The first sample is left out: it is taken as the burners start, so it can still read idle
    clocks, which run above the busy cores' all-core turbo.

    >>> rows = ((0, 3900, 300), (5, 3000, 300), (60, 3000, 2800), (120, 2400, 2800), (180, 2400, 2800), (200, 3000, 700))
    >>> samples = [{"t_s": t, "busy_mhz": mhz, "gpu_w": w} for t, mhz, w in rows]
    >>> summarize(samples, CpuLoad(min_full_power_s=15))
    {'max_mhz': 3000, 'full_power_samples': 3, 'full_power_mhz': 2600, 'low_share': 0.667}
    >>> summarize(samples, CpuLoad(min_full_power_s=20))
    {'max_mhz': 3000, 'full_power_samples': 3, 'full_power_mhz': None, 'low_share': None}
    """
    samples = samples[1:]
    top = max((sample["busy_mhz"] or 0 for sample in samples), default=0)
    peak = max((sample["gpu_w"] or 0 for sample in samples), default=0)
    loaded = [sample for sample in samples if peak and (sample["gpu_w"] or 0) >= cfg.full_power_frac * peak]
    if not top or len(loaded) * cfg.interval_s < cfg.min_full_power_s:
        return {"max_mhz": top, "full_power_samples": len(loaded), "full_power_mhz": None, "low_share": None}
    low = [sample for sample in loaded if sample["busy_mhz"] < (1 - cfg.low_frac) * top]
    return {
        "max_mhz": top,
        "full_power_samples": len(loaded),
        "full_power_mhz": _mean([sample["busy_mhz"] for sample in loaded]),
        "low_share": round(len(low) / len(loaded), 3),
    }


class HealthCheck(HealthCheckBase):
    def __init__(self, cfg: CpuLoad, gpus_done: threading.Event, cpus: list[int]) -> None:
        self.cfg = cfg
        self.gpus_done = gpus_done
        self.cpus = cpus

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
        """Hold the load until gpus_done is set; cpus are the job's, os.sched_getaffinity(0).

        The burners run at SCHED_IDLE, so the other phase A steps, DCGM and the Xid watcher never wait for a CPU.
        They stop when dcgm returns; its own timeout ends a hang.
        """
        busy = spread(sorted(self.cpus), self.cfg.busy_frac)
        throttles_before = throttle_count()
        burners = []
        samples = []
        start = time.monotonic()
        try:
            for cpu in busy:
                burners.append(start_burner(cpu))
            with contextlib.ExitStack() as nvml:
                with contextlib.suppress(pynvml.NVMLError):  # gpu_w is None without a driver
                    nvml.enter_context(session())
                while not self.gpus_done.is_set():
                    mhz = cpu_mhz()
                    samples.append(
                        {
                            "t_s": round(time.monotonic() - start, 1),
                            "busy_mhz": _mean([mhz[cpu] for cpu in busy if cpu in mhz]),
                            "idle_mhz": _mean([mhz[cpu] for cpu in self.cpus if cpu in mhz and cpu not in busy]),
                            "gpu_w": watts(),
                        }
                    )
                    self.gpus_done.wait(self.cfg.interval_s)
        finally:
            for burner in burners:
                burner.kill()
            for burner in burners:
                burner.wait()
        held_s = round(time.monotonic() - start, 1)
        if held_s < self.cfg.min_full_power_s:  # dcgm skipped or failed early: too short to judge
            return {"status": Status.SKIPPED, "held_s": held_s, "reason": f"dcgm returned after {held_s} s"}
        return {
            "status": Status.OK,
            "busy_cpus": cpulist(busy),
            "held_s": held_s,
            **summarize(samples, self.cfg),
            "throttle_delta": throttle_count() - throttles_before,
            "samples": samples,
        }

    @staticmethod
    def report(results: ByHost[JsonDict], cfg: CpuLoad, name: str) -> Report:
        findings = per_host(results, cfg, name, _findings)
        mhz = {
            host: data["full_power_mhz"]
            for host, data in results.items()
            if data.get("status") != Status.SKIPPED
            and data.get("full_power_mhz")
        }
        if not mhz:
            return Report(findings, {})
        mid, floor, slow = below_median(mhz, cfg.min_frac_of_median)
        for host in slow:
            detail = f"busy-core clock {mhz[host]} < {floor:.0f} MHz"
            findings.setdefault(host, []).append(Finding("warn", "cpu_load.median", detail))
        return Report(findings, {"cpu_load_busy_mhz": mid})


def _findings(data: JsonDict, cfg: CpuLoad) -> Iterator[Finding]:
    low_share = data.get("low_share") or 0
    if low_share > cfg.fail_low_share:
        severity: Severity = "fail"
    elif low_share > cfg.warn_low_share:
        severity = "warn"
    else:
        severity = None
    if severity:
        detail = (
            f"{low_share:.0%} of busy-core readings at full GPU power over {cfg.low_frac:.0%} below "
            f"{data['max_mhz']} MHz (mean {data['full_power_mhz']})"
        )
        yield Finding(severity, "cpu_load.low_clocks", detail)
    if data["throttle_delta"]:
        yield Finding("warn", "cpu_load.throttle", f"{data['throttle_delta']} CPU thermal throttle events")
