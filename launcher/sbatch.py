"""Submit sbatch job scripts."""

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple


class SbatchResult(NamedTuple):
    job_id: int | None  # None if the submission failed
    returncode: int
    stdout: str
    stderr: str


def sbatch(script: str, options: Sequence[str], cwd: Path) -> SbatchResult:
    """Submit a job script to SLURM; options override the script's #SBATCH directives."""
    try:
        # cwd: job_submit.lua may refuse jobs submitted from /home
        result = subprocess.run(
            ["sbatch", "--parsable", *options], input=script, capture_output=True, text=True, cwd=cwd
        )
    except FileNotFoundError as error:  # also raised for a missing cwd
        result = subprocess.CompletedProcess([], 127, "", f"Must be run on Slurm ({error.filename} not found).")
    try:
        job_id = int(result.stdout.split(";")[0])
    except ValueError:
        job_id = None
    return SbatchResult(job_id, result.returncode, result.stdout, result.stderr)
