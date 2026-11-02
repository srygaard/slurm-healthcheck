"""The HealthCheckBase, the findings a check reports, and the runner that records a result."""

import statistics
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal, NamedTuple

from healthcheck.util.file import Json, JsonDict
from healthcheck.util.hostlist import ByHost
from healthcheck.util.status import Status, ok

type Severity = Literal["fail", "warn"]


class Finding(NamedTuple):
    severity: Severity
    check: str
    detail: str


class Report(NamedTuple):
    """findings is by host. medians omits a key when no node had data for it."""

    findings: ByHost[list[Finding]]
    medians: dict[str, float | None]


def median(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(statistics.median(present), 4) if present else None


def below_median(values_by_host: ByHost[float], frac: float) -> tuple[float | None, float, list[str]]:
    """The median, frac of it, and the hosts below that floor, in insertion order."""
    mid = median(values_by_host.values())
    floor = (mid or 0) * frac
    return mid, floor, [
        host
        for host, value in values_by_host.items()
        if value < floor
    ]


def per_host[S](
    results: ByHost[JsonDict],
    cfg: S,
    name: str,
    findings: Callable[[JsonDict, S], Iterable[Finding]],
    on_skipped: Callable[[JsonDict, S, str], Iterable[Finding]] | None = None,
) -> ByHost[list[Finding]]:
    """Per-host findings. A skipped step is silent unless on_skipped says otherwise."""
    out: ByHost[list[Finding]] = {}
    for host, data in results.items():
        if data.get("status") == Status.SKIPPED:
            found = list(on_skipped(data, cfg, name)) if on_skipped else []
        else:
            found = list(findings(data, cfg))
        if found:
            out[host] = found
    return out


class Availability(NamedTuple):
    """Whether the required tool is present. reason is None when available is True."""

    available: bool
    reason: str | None = None


class HealthCheckBase:
    """A node step. Subclasses run it and turn its results into findings.

    report() stays on the subclass: its arguments are that check's config section.
    """

    @property
    def enabled(self) -> bool:
        """False when this check's config turns it off. Checks with no flag are always True."""
        raise NotImplementedError

    def available(self) -> Availability:
        """Whether the required tool is present. Does not read enabled."""
        raise NotImplementedError

    def collect(self) -> JsonDict:
        """Run the check. The runner calls this only when the check is enabled and available."""
        raise NotImplementedError

    @classmethod
    def missing(cls, name: str) -> Finding | None:
        """A finding when the node's JSON has no such step. None means that is fine."""
        return Finding("fail", name, "not run")

    @classmethod
    def unwrap(cls, node: dict[str, Any], name: str) -> tuple[Any, Finding | None]:
        """The step's data, and a finding when the step is missing, unavailable, or failed.

        Data whose own status is skipped is returned for the check to judge.
        """
        result = node.get(name)
        if result is None:
            return None, cls.missing(name)
        if result.get("status") == Status.UNAVAILABLE:
            return None, Finding("warn", name, f"{result['status']}: {result.get('error', '')}".strip(": "))
        if not ok(result):
            return None, Finding("fail", name, f"{result['status']}: {result.get('error', '')}".strip(": "))
        return result["data"], None


def step(check: HealthCheckBase) -> JsonDict:
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
