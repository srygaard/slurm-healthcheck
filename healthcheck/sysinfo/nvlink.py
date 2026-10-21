"""NVLink state and error counters of one GPU, from NVML."""

import enum
from typing import Any

import pynvml

from healthcheck.util.nvml import query


class NvlinkError(enum.IntEnum):
    REPLAY = pynvml.NVML_NVLINK_ERROR_DL_REPLAY
    RECOVERY = pynvml.NVML_NVLINK_ERROR_DL_RECOVERY
    CRC_FLIT = pynvml.NVML_NVLINK_ERROR_DL_CRC_FLIT
    CRC_DATA = pynvml.NVML_NVLINK_ERROR_DL_CRC_DATA


def active(handle: Any) -> list[int]:
    return [
        link
        for link in range(pynvml.NVML_NVLINK_MAX_LINKS)
        if query(pynvml.nvmlDeviceGetNvLinkState, handle, link) == pynvml.NVML_FEATURE_ENABLED
    ]


def errors(handle: Any) -> dict[str, int]:
    """Error counters summed over active links."""
    links = active(handle)
    return {
        error.name.lower(): sum(
            query(pynvml.nvmlDeviceGetNvLinkErrorCounter, handle, link, error) or 0 for link in links
        )
        for error in NvlinkError
    }
