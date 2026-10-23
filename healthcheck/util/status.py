"""The status of a step or test, as written to its JSON; a StrEnum, so a value read back compares equal to its member."""

import enum
from collections.abc import Mapping
from typing import Any


class Status(enum.StrEnum):
    OK = "ok"
    ERROR = "error"  # the step crashed, or an NCCL test exited nonzero or printed no bandwidth
    UNAVAILABLE = "unavailable"  # the tool or permission is missing: not a node fault
    MISSING = "missing"  # an input setup should have written
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    WRONG = "wrong"  # NCCL's result check found wrong values


def ok(result: Mapping[str, Any]) -> bool:
    """
    Whether a step's or test's result, fresh or read back from JSON, has status ok.

    >>> ok({"status": "ok", "data": {}}), ok({"status": Status.TIMEOUT}), ok({})
    (True, False, False)
    """
    return result.get("status") == Status.OK
