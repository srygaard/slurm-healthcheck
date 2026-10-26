"""Step 3a: single-node DCGM diagnostics with the host install (see probe)."""

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from healthcheck.config import Dcgm
from healthcheck.util.cmd import Cmd, run
from healthcheck.util.file import JsonDict
from healthcheck.util.status import Status

# Our own hostengine, for hosts without a running one: non-root, so keep pid and log in a temp dir.
SELF_HOSTED = """
nv-hostengine -n --port {port} --pid {tmp}/he.pid --log-filename {tmp}/he.log & he=$!
for _ in $(seq 40); do dcgmi discovery --host localhost:{port} -l >/dev/null 2>&1 && break; sleep 0.5; done
dcgmi diag --host localhost:{port} {args}; rc=$?
kill $he; wait $he 2>/dev/null
exit $rc
"""


class DiagError(RuntimeError):
    """dcgmi diag printed no test results; out holds what it printed."""

    def __init__(self, out: Cmd):
        detail = _runtime_error(out) or (out.stderr or out.stdout)[-500:].strip()
        super().__init__(f"dcgmi diag exited {out.returncode}: {detail}")
        self.out = out


def _runtime_error(out: Cmd) -> str | None:
    """The "runtime_error" of a `dcgmi diag -j` that refused to run, on one line."""
    try:
        report = json.loads(out.stdout[out.stdout.index("{") :])
    except ValueError:
        return None
    error = report.get("DCGM Diagnostic", {}).get("runtime_error") if isinstance(report, dict) else None
    return " ".join(error.split()) if isinstance(error, str) else None


def command(mode: str, args: list[str], port: int, tmp: str) -> list[str]:
    """mode is "host" (running hostengine) or "host-self" (start one)."""
    if mode == "host":
        return ["dcgmi", "diag", *args]
    return ["bash", "-c", SELF_HOSTED.format(port=port, tmp=tmp, args=shlex.join(args))]


def diag(
    mode: str, suite: str, port: int, timeout: float | None, log_dir: Path, parameters: str = ""
) -> dict[str, Any]:
    """Run and parse `dcgmi diag`; suite is a level ("1"-"4") or comma-separated test names, and parameters is
    the `-p` argument, ";"-separated.

    dcgmi exits nonzero when a test fails, so the exit code alone isn't an error: only a run that reported no tests is.
    """
    # DCGM 4 refuses to run when dcgmi and the hostengine see different CUDA_VISIBLE_DEVICES, and Slurm sets it
    # for the job. Jobs hold whole nodes, so testing every GPU is what we want anyway.
    env = {name: value for name, value in os.environ.items() if name != "CUDA_VISIBLE_DEVICES"}
    with tempfile.TemporaryDirectory(prefix="hc-dcgm-", dir=os.environ.get("SLURM_TMPDIR")) as tmp:
        args = ["-r", suite, "-j"]
        if mode == "host-self":
            # Our hostengine runs as us and logs to the working directory, which all nodes of a job share. The host's
            # runs as another user that can't write to our directories, and logs outside them anyway.
            log_dir.mkdir(parents=True, exist_ok=True)
            args += ["--debugLogFile", str(log_dir / "nvvs.log"), "--statspath", str(log_dir)]
        args += ["-p", parameters] if parameters else []
        out = run(command(mode, args, port, tmp), timeout=timeout, env=env)
    try:
        tests = parse(out.stdout)["tests"]
    except ValueError:  # no JSON at all
        tests = {}
    if not tests:
        raise DiagError(out)
    return {"returncode": out.returncode, "tests": tests}


def _find_categories(report: Any) -> list[dict[str, Any]]:
    """The "test_categories" list, wherever this DCGM version nests it."""
    if isinstance(report, dict):
        if "test_categories" in report:
            return report["test_categories"]
        for value in report.values():
            if found := _find_categories(value):
                return found
    return []


def parse(stdout: str) -> dict[str, Any]:
    """Reduce `dcgmi diag -j` output to the worst status per test, with per-GPU statuses and messages."""
    severity = {"Fail": 3, "Warn": 2, "Pass": 1}  # anything else (Skip, Not Run) ranks 0
    report = json.loads(stdout[stdout.index("{") :])  # some versions print a banner first
    tests = {}
    for category in _find_categories(report):
        for test in category.get("tests", []):
            entities = {}
            messages = []
            for result in test.get("results", []):
                entity = result.get("entity_id", result.get("gpu_ids", "all"))
                entities[str(entity)] = result.get("status", "Unknown")
                for key in ("warnings", "info", "errors"):
                    for msg in result.get(key) or []:
                        messages.append(msg.get("warning") or msg.get("message") if isinstance(msg, dict) else msg)
            statuses = list(entities.values()) or [test.get("test_summary", {}).get("status", "Unknown")]
            tests[test["name"]] = {
                "category": category.get("category"),
                "status": max(statuses, key=lambda status: severity.get(status, 0)),
                "entities": entities,
                "messages": [message for message in messages if message][:20],
            }
    return {"tests": tests}


def collect(cfg: Dcgm, env: dict[str, Any], log_dir: Path) -> JsonDict:
    mode = env.get("dcgm", "none")
    if not cfg.enabled or mode == "none":
        return {"status": Status.SKIPPED, "mode": mode}
    tests = ",".join(cfg.tests) or str(
        cfg.level
    )  # the `dcgmi diag -r` argument: the listed tests if any, else the level
    return {
        "status": Status.OK,
        "mode": mode,
        "suite": tests,
        **diag(mode, tests, cfg.port, None, log_dir, parameters(cfg)),
    }


def parameters(cfg: Dcgm) -> str:
    """
    The `dcgmi diag -p` argument: cfg.parameters and diagnostic's precision.

    >>> parameters(Dcgm(parameters=["diagnostic.test_duration=30"], diagnostic_precision="fp16"))
    'diagnostic.test_duration=30;diagnostic.precision=h'
    """
    return ";".join([*cfg.parameters, f"diagnostic.precision={cfg.diagnostic_precision_letter}"])
