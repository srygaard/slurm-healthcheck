"""GPU configuration and error counters per device, from NVML, plus the nvidia-smi topology matrix."""

import enum
import hashlib
import re
from typing import Any

import pynvml

from healthcheck.sysinfo import nvlink
from healthcheck.util.cmd import run
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


def counters() -> list[JsonDict]:
    """Error counters per GPU; finalize compares a second snapshot against the one from step 1."""
    out: list[JsonDict] = []
    with session():
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            throttle = HwThrottle(_throttle_reasons(handle) or 0)
            out.append(
                {
                    "index": index,
                    "ecc_volatile": _ecc_errors(handle, pynvml.NVML_VOLATILE_ECC),
                    "ecc_aggregate": _ecc_errors(handle, pynvml.NVML_AGGREGATE_ECC),
                    "remapped_rows": _remapped_rows(handle),
                    "nvlink_errors": nvlink.errors(handle),
                    # Unknown bits are ignored, so this lists only the hardware reasons
                    "hw_throttle": [
                        name.lower() for name, reason in HwThrottle.__members__.items() if reason in throttle
                    ],
                }
            )
    return out


def info() -> JsonDict:
    with session():
        cuda = query(pynvml.nvmlSystemGetCudaDriverVersion_v2)
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            pci = query(pynvml.nvmlDeviceGetPciInfo, handle)
            ecc = query(pynvml.nvmlDeviceGetEccMode, handle)
            mig = query(pynvml.nvmlDeviceGetMigMode, handle)
            memory = query(pynvml.nvmlDeviceGetMemoryInfo, handle)
            power = query(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
            devices.append(
                {
                    "index": index,
                    "name": query(pynvml.nvmlDeviceGetName, handle),
                    "uuid": query(pynvml.nvmlDeviceGetUUID, handle),
                    "bus_id": (pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId) if pci else None,
                    "vbios": query(pynvml.nvmlDeviceGetVbiosVersion, handle),
                    "memory_gib": round(memory.total / 2**30, 1) if memory else None,
                    "ecc_enabled": bool(ecc[0]) if ecc else None,
                    "mig_enabled": bool(mig[0]) if mig else False,
                    "persistence": query(pynvml.nvmlDeviceGetPersistenceMode, handle),
                    "pcie_gen": query(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle),
                    "pcie_gen_max": query(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle),
                    "pcie_width": query(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle),
                    "pcie_width_max": query(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle),
                    "power_limit_w": power / 1000 if power else None,
                    "max_sm_clock_mhz": query(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_SM),
                    "max_mem_clock_mhz": query(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_MEM),
                    "temperature_c": query(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU),
                    "nvlinks_active": len(nvlink.active(handle)),
                }
            )
        info = {
            "count": len(devices),
            "driver_version": query(pynvml.nvmlSystemGetDriverVersion),
            "cuda_version": f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None,
            "devices": devices,
        }

    # The GPU/NIC/CPU affinity matrix should be identical on identical hardware
    topo = run(["nvidia-smi", "topo", "-m"], timeout=20)
    if topo.returncode == 0:
        text = re.sub(r"\x1b\[[0-9;]*m", "", topo.stdout).strip()
        info["topo_hash"] = hashlib.sha256(text.encode()).hexdigest()[:12]
        info["topo"] = text
    return info
