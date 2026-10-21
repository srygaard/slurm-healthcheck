"""NVML session and queries that tolerate unsupported devices and drivers."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pynvml


@contextmanager
def session() -> Generator[None]:
    """NVML initialized for the block; init is reference counted, so nesting is fine."""
    pynvml.nvmlInit()
    try:
        yield
    finally:
        pynvml.nvmlShutdown()


def query(fn: Callable[..., Any], *args: Any) -> Any:
    """Call an NVML query, None if the device or driver doesn't support it."""
    try:
        value = fn(*args)
    except pynvml.NVMLError:
        return None
    return value.decode() if isinstance(value, bytes) else value
