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


@contextmanager
def event_set() -> Generator[Any]:
    """An NVML event set. Enter it inside session(), so free runs before nvmlShutdown."""
    events = pynvml.nvmlEventSetCreate()
    try:
        yield events
    finally:
        pynvml.nvmlEventSetFree(events)


def query(fn: Callable[..., Any], *args: Any) -> Any:
    """Call an NVML query, None if the device or driver doesn't support it."""
    try:
        value = fn(*args)
    except pynvml.NVMLError:
        return None
    return value.decode() if isinstance(value, bytes) else value
