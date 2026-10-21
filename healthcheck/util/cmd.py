"""Run a command and capture its output, without raising."""

import subprocess
from collections.abc import Sequence
from typing import Any, NamedTuple


class Cmd(NamedTuple):
    returncode: int  # 124 on timeout, 127 if the binary is missing
    stdout: str
    stderr: str


def run(cmd: Sequence[str], timeout: float | None, **kwargs: Any) -> Cmd:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
    except FileNotFoundError as error:
        return Cmd(127, "", str(error))
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        return Cmd(124, stdout, f"timeout after {timeout}s")
    return Cmd(completed.returncode, completed.stdout, completed.stderr)
