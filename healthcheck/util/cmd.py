"""Run a command and capture its output, without raising."""

import enum
import subprocess
from collections.abc import Sequence
from typing import Any, NamedTuple


class ExitCode(enum.IntEnum):
    """Codes assigned here when the process did not exit on its own.

    Python does not name them. GNU timeout uses 124. A shell uses 127 when the command is missing.
    """

    TIMEOUT = 124
    NOT_FOUND = 127


class Cmd(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def run(cmd: Sequence[str], timeout: float | None, **kwargs: Any) -> Cmd:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **kwargs)
    except FileNotFoundError as error:
        return Cmd(ExitCode.NOT_FOUND, "", str(error))
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        return Cmd(ExitCode.TIMEOUT, stdout, f"timeout after {timeout}s")
    return Cmd(completed.returncode, completed.stdout, completed.stderr)
