"""GPU configuration and error counters per device, from NVML, plus the nvidia-smi topology matrix."""

import enum
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self

import pynvml

from healthcheck.sysinfo import nvlink
from healthcheck.util.cmd import run
from healthcheck.util.device import BusId, PciDevice
from healthcheck.util.file import JsonDict
from healthcheck.util.nvml import query, session


class HwThrottle(enum.IntFlag):
    """Clock event (throttle) reasons that point at hardware rather than workload."""

    HW_SLOWDOWN = pynvml.nvmlClocksThrottleReasonHwSlowdown
    SW_THERMAL = pynvml.nvmlClocksThrottleReasonSwThermalSlowdown
    HW_THERMAL = pynvml.nvmlClocksThrottleReasonHwThermalSlowdown
    HW_POWER_BRAKE = pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown


def _throttle_reasons(handle: Any) -> int | None:
    fn = (
        getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
        or pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
    )
    return query(fn, handle)


def _ecc_errors(handle: Any, counter: int) -> dict[str, int | None]:
    return {
        "uncorrected": query(
            pynvml.nvmlDeviceGetTotalEccErrors, handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, counter
        ),
        "corrected": query(
            pynvml.nvmlDeviceGetTotalEccErrors, handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, counter
        ),
    }


def _remapped_rows(handle: Any) -> dict[str, int] | None:
    if (rows := query(pynvml.nvmlDeviceGetRemappedRows, handle)) is None:
        return None
    corrected, uncorrected, pending, failure = rows
    return {"corrected": corrected, "uncorrected": uncorrected, "pending": int(pending), "failure": int(failure)}


def _handles() -> list[tuple[int, Any]]:
    """Each GPU's index and NVML handle; call inside session()."""
    return [(index, pynvml.nvmlDeviceGetHandleByIndex(index)) for index in range(pynvml.nvmlDeviceGetCount())]


def watts() -> float | None:
    """Total power draw in watts; None if any GPU can't be read. Call inside session().

    One unsupported device drops the whole reading: a partial sum would look like a lighter load.
    """
    try:
        return round(sum(pynvml.nvmlDeviceGetPowerUsage(handle) for _, handle in _handles()) / 1000)
    except pynvml.NVMLError:
        return None


class PowerLimits(NamedTuple):
    """A GPU's power limits in watts. Floats, so finalize's counter deltas leave them out."""

    enforced_w: float | None  # the lowest of every limit set: by an admin, the BMC or the driver
    default_w: float | None  # the board's default

    def power_capped(self, frac: float) -> bool:
        """
        Whether the enforced limit is under frac of the default; False when either can't be read.

        >>> PowerLimits(enforced_w=200.0, default_w=1000.0).power_capped(0.95)
        True
        >>> PowerLimits(enforced_w=None, default_w=1000.0).power_capped(0.95)
        False
        """
        return bool(self.enforced_w and self.default_w and self.enforced_w < frac * self.default_w)


def _power_limits(handle: Any) -> PowerLimits:
    enforced = query(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
    default = query(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handle)
    return PowerLimits(
        enforced_w=enforced / 1000 if enforced else None,
        default_w=default / 1000 if default else None,
    )


@dataclass(frozen=True, kw_only=True)
class GpuState:
    """One GPU's error counters and power limits at a point in time; finalize compares a second snapshot against
    sysinfo's."""

    index: int
    ecc_volatile: dict[str, int | None]
    ecc_aggregate: dict[str, int | None]
    remapped_rows: dict[str, int] | None
    nvlink_errors: dict[str, int]
    hw_throttle: list[str]  # Unknown bits are ignored, so this lists only the hardware reasons
    power_limit_w: float | None  # enforced
    default_power_limit_w: float | None

    @classmethod
    def from_nvml(cls, index: int, handle: Any) -> Self:
        throttle = HwThrottle(_throttle_reasons(handle) or 0)
        limits = _power_limits(handle)
        return cls(
            index=index,
            ecc_volatile=_ecc_errors(handle, pynvml.NVML_VOLATILE_ECC),
            ecc_aggregate=_ecc_errors(handle, pynvml.NVML_AGGREGATE_ECC),
            remapped_rows=_remapped_rows(handle),
            nvlink_errors=nvlink.errors(handle),
            hw_throttle=[name.lower() for name, reason in HwThrottle.__members__.items() if reason in throttle],
            power_limit_w=limits.enforced_w,
            default_power_limit_w=limits.default_w,
        )

    @property
    def power_limits(self) -> PowerLimits:
        return PowerLimits(enforced_w=self.power_limit_w, default_w=self.default_power_limit_w)

    def to_json(self) -> JsonDict:
        return asdict(self)


def states() -> list[GpuState]:
    with session():
        return [GpuState.from_nvml(index, handle) for index, handle in _handles()]


@dataclass(frozen=True, kw_only=True)
class Gpu(PciDevice):
    """One NVML device; name is its model, e.g. "NVIDIA H100 80GB HBM3"."""

    index: int
    uuid: str | None
    vbios: str | None
    memory_gib: float | None
    ecc_enabled: bool | None
    mig_enabled: bool
    persistence: int | None
    pcie_gen: int | None
    pcie_gen_max: int | None
    pcie_width: int | None
    pcie_width_max: int | None
    max_sm_clock_mhz: int | None
    max_mem_clock_mhz: int | None
    temperature_c: int | None
    nvlinks_active: int

    @classmethod
    def from_nvml(cls, index: int, handle: Any, root: Path = Path("/sys")) -> Self:
        """NVML's fields, and the PCI ones from sysfs at NVML's bus id."""
        pci = query(pynvml.nvmlDeviceGetPciInfo, handle)
        ecc = query(pynvml.nvmlDeviceGetEccMode, handle)
        mig = query(pynvml.nvmlDeviceGetMigMode, handle)
        memory = query(pynvml.nvmlDeviceGetMemoryInfo, handle)
        bus_id = pci.busId if pci else None
        return cls.from_sysfs(
            BusId.parse(bus_id.decode() if isinstance(bus_id, bytes) else bus_id),
            root,
            index=index,
            name=query(pynvml.nvmlDeviceGetName, handle),
            uuid=query(pynvml.nvmlDeviceGetUUID, handle),
            vbios=query(pynvml.nvmlDeviceGetVbiosVersion, handle),
            memory_gib=round(memory.total / 2**30, 1) if memory else None,
            ecc_enabled=bool(ecc[0]) if ecc else None,
            mig_enabled=bool(mig[0]) if mig else False,
            persistence=query(pynvml.nvmlDeviceGetPersistenceMode, handle),
            pcie_gen=query(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle),
            pcie_gen_max=query(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle),
            pcie_width=query(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle),
            pcie_width_max=query(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle),
            max_sm_clock_mhz=query(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_SM),
            max_mem_clock_mhz=query(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_MEM),
            temperature_c=query(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU),
            nvlinks_active=len(nvlink.active(handle)),
        )


def devices(root: Path = Path("/sys")) -> list[Gpu]:
    with session():
        return [Gpu.from_nvml(index, handle, root) for index, handle in _handles()]


def info() -> JsonDict:
    """Driver and CUDA versions and the topology matrix; sysinfo adds the devices."""
    with session():
        cuda = query(pynvml.nvmlSystemGetCudaDriverVersion_v2)
        info = {
            "driver_version": query(pynvml.nvmlSystemGetDriverVersion),
            "cuda_version": f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None,
        }

    # The GPU/NIC/CPU affinity matrix should be identical on identical hardware
    topo = run(["nvidia-smi", "topo", "-m"], timeout=20)
    if topo.returncode == 0:
        text = re.sub(r"\x1b\[[0-9;]*m", "", topo.stdout).strip()
        info["topo_hash"] = hashlib.sha256(text.encode()).hexdigest()[:12]
        info["topo"] = text
    return info
