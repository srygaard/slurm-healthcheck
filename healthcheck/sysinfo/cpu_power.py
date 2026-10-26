"""The CPU clock and power settings a regular user can read: cpufreq, turbo and RAPL limits.

The settings that decide them at runtime (BIOS power profile, C-states, BMC capping) need root or the BMC.
"""

from pathlib import Path

from healthcheck.util.file import JsonDict, read_sys


def _distinct(paths: list[Path]) -> list[str]:
    """The values the files hold, each once: one value when every CPU agrees."""
    return sorted({value for path in paths if (value := read_sys(path)) is not None})


def _mhz(khz: list[str]) -> list[int]:
    """
    >>> _mhz(["4100000", "800000"])
    [800, 4100]
    """
    return sorted(int(value) // 1000 for value in khz)


def cpufreq(cpus: Path) -> JsonDict:
    """Per-CPU settings under cpus (/sys/devices/system/cpu), as the distinct values across CPUs."""
    dirs = sorted(cpus.glob("cpu[0-9]*/cpufreq"))
    return {
        "driver": _distinct([path / "scaling_driver" for path in dirs]),
        "governor": _distinct([path / "scaling_governor" for path in dirs]),
        "max_mhz": _mhz(_distinct([path / "cpuinfo_max_freq" for path in dirs])),
        "scaling_max_mhz": _mhz(_distinct([path / "scaling_max_freq" for path in dirs])),
        "epp": _distinct([path / "energy_performance_preference" for path in dirs]),
    }


def turbo(cpus: Path) -> JsonDict:
    """intel_pstate's turbo switch and limits, or the generic boost switch acpi-cpufreq and amd-pstate use."""
    pstate = cpus / "intel_pstate"
    return {
        "intel_pstate": read_sys(pstate / "status"),
        "no_turbo": read_sys(pstate / "no_turbo"),
        "max_perf_pct": read_sys(pstate / "max_perf_pct"),
        "min_perf_pct": read_sys(pstate / "min_perf_pct"),
        "boost": read_sys(cpus / "cpufreq" / "boost"),
    }


def rapl(powercap: Path) -> list[JsonDict]:
    """Each RAPL zone (package-0, its dram subzone, psys) with its power limits in W by constraint name."""
    zones = []
    for zone in sorted(powercap.glob("intel-rapl:*")):
        limits = {}
        for name in sorted(zone.glob("constraint_*_name")):
            microwatts = read_sys(zone / name.name.replace("_name", "_power_limit_uw"))
            limits[read_sys(name) or name.name] = int(microwatts) / 1e6 if microwatts else None
        zones.append(
            {
                "zone": zone.name,
                "name": read_sys(zone / "name"),
                "enabled": read_sys(zone / "enabled"),
                "limits_w": limits,
            }
        )
    return zones


def collect(root: Path = Path("/sys")) -> JsonDict:
    cpus = root / "devices" / "system" / "cpu"
    return {**cpufreq(cpus), **turbo(cpus), "rapl": rapl(root / "class" / "powercap")}
