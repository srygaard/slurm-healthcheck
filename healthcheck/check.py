"""The Check protocol and the runner that records a skipped, unavailable, or collected result."""

import time
from typing import NamedTuple, Protocol

from healthcheck.util.file import Json, JsonDict
from healthcheck.util.status import Status


class Availability(NamedTuple):
    """Whether the required tool is present. reason is None when available is True."""

    available: bool
    reason: str | None = None


class Check(Protocol):
    @property
    def enabled(self) -> bool:
        """False when this check's config turns it off. Checks with no flag are always True."""

    def available(self) -> Availability:
        """Whether the required tool is present. Does not read enabled."""

    def collect(self) -> JsonDict:
        """Run the check. The runner calls this only when the check is enabled and available."""


def step(check: Check) -> JsonDict:
    """Run one check; a crash is recorded rather than propagated, so other checks still report.

    A disabled check is skipped without probing. A missing tool is unavailable, not a failure.
    """
    start = time.monotonic()
    try:
        if not check.enabled:
            result: dict[str, Json] = {"status": Status.OK, "data": {"status": Status.SKIPPED}}
        else:
            found = check.available()
            if not found.available:
                result = {"status": Status.UNAVAILABLE, "error": found.reason or ""}
            else:
                result = {"status": Status.OK, "data": check.collect()}
    except Exception as error:  # noqa: BLE001 - any failure is a node finding
        result = {"status": Status.ERROR, "error": f"{type(error).__name__}: {error}"}
    result["elapsed_s"] = round(time.monotonic() - start, 2)
    return result
