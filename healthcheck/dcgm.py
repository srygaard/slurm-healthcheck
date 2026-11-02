"""Step 3a: single-node DCGM diagnostics with the host install."""

import enum
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from healthcheck.check import Availability, Finding, HealthCheckBase, Report, Severity, median, per_host
from healthcheck.config import Dcgm
from healthcheck.util.cmd import Cmd, ExitCode, run
from healthcheck.util.file import JsonDict
from healthcheck.util.hostlist import ByHost
from healthcheck.util.status import Status

# Our own hostengine, for hosts without a running one: non-root, so keep pid and log in a temp dir.
SELF_HOSTED = """
nv-hostengine -n --port {port} --pid {tmp}/he.pid --log-filename {tmp}/he.log & he=$!
for _ in $(seq 40); do dcgmi discovery --host localhost:{port} -l >/dev/null 2>&1 && break; sleep 0.5; done
dcgmi diag --host localhost:{port} {args}; rc=$?
kill $he; wait $he 2>/dev/null
exit $rc
"""


class Mode(enum.StrEnum):
    """How dcgmi reaches a hostengine."""

    NONE = "none"  # dcgmi is missing, or no engine is running and we cannot start one
    HOST = "host"  # a hostengine is already running
    HOST_SELF = "host-self"  # start nv-hostengine for this run


class DiagError(RuntimeError):
    """dcgmi diag printed no test results."""


def _runtime_error(out: Cmd) -> str | None:
    """The "runtime_error" of a `dcgmi diag -j` that refused to run, on one line."""
    try:
        report = json.loads(out.stdout[out.stdout.index("{") :])
    except ValueError:
        return None
    error = report.get("DCGM Diagnostic", {}).get("runtime_error") if isinstance(report, dict) else None
    return " ".join(error.split()) if isinstance(error, str) else None


def run_hostengine(cmd: list[str], env: dict[str, str], timeout_s: float) -> Cmd:
    """Run the self-hosted script in its own session, so a timeout kills nv-hostengine as well as the shell."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, env=env
        )
    except FileNotFoundError as error:
        return Cmd(ExitCode.NOT_FOUND, "", str(error))
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return Cmd(ExitCode.TIMEOUT, stdout or "", f"timeout after {timeout_s}s")
    return Cmd(proc.returncode if proc.returncode is not None else 0, stdout, stderr)


def command(mode: Mode, args: list[str], port: int, tmp: str) -> list[str]:
    """The dcgmi diag command for a running hostengine, or a shell that starts one."""
    if mode == Mode.HOST:
        return ["dcgmi", "diag", *args]
    return ["bash", "-c", SELF_HOSTED.format(port=port, tmp=tmp, args=shlex.join(args))]


def choose_mode(discovery_timeout_s: float) -> Mode:
    """The running hostengine, else one we start, else none.

    discovery only checks that a hostengine answers.
    """
    if not shutil.which("dcgmi"):
        return Mode.NONE
    if run(["dcgmi", "discovery", "-l"], timeout=discovery_timeout_s).returncode == 0:
        return Mode.HOST
    if shutil.which("nv-hostengine"):
        return Mode.HOST_SELF
    return Mode.NONE


def diag(mode: Mode, suite: str, port: int, log_dir: Path, timeout_s: float, parameters: str = "") -> dict[str, Any]:
    """Run and parse `dcgmi diag`; suite is a level ("1"-"4") or comma-separated test names, and parameters is
    the `-p` argument, ";"-separated.

    dcgmi exits nonzero when a test fails, so the exit code alone isn't an error: only a run that reported no tests is.
    """
    # DCGM 4 refuses to run when dcgmi and the hostengine see different CUDA_VISIBLE_DEVICES, and Slurm sets it
    # for the job. Jobs hold whole nodes, so testing every GPU is what we want anyway.
    env = {name: value for name, value in os.environ.items() if name != "CUDA_VISIBLE_DEVICES"}
    with tempfile.TemporaryDirectory(prefix="hc-dcgm-", dir=os.environ.get("SLURM_TMPDIR")) as tmp:
        # DCGM 4.4.2+: the hostengine stops nvvs when this client dies. Without it, a cancelled job
        # leaves the diagnostic holding the GPUs and the next dcgmi diag exits 222 (resource in use).
        args = ["-r", suite, "-j", "--enable-heartbeat"]
        if mode == Mode.HOST_SELF:
            # Our hostengine runs as us and logs to the working directory, which all nodes of a job share. The host's
            # runs as another user that can't write to our directories, and logs outside them anyway.
            log_dir.mkdir(parents=True, exist_ok=True)
            args += ["--debugLogFile", str(log_dir / "nvvs.log"), "--statspath", str(log_dir)]
        args += ["-p", parameters] if parameters else []
        cmd = command(mode, args, port, tmp)
        # The shell backgrounds nv-hostengine, so a plain timeout would kill the shell and leave the engine.
        out = run_hostengine(cmd, env, timeout_s) if mode == Mode.HOST_SELF else run(cmd, timeout=timeout_s, env=env)
    try:
        tests = parse(out.stdout)["tests"]
    except ValueError:  # no JSON at all
        tests = {}
    if out.returncode == ExitCode.TIMEOUT or not tests:
        detail = _runtime_error(out) or (out.stderr or out.stdout)[-500:].strip()
        raise DiagError(f"dcgmi diag exited {out.returncode}: {detail}")
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


class HealthCheck(HealthCheckBase):
    def __init__(self, cfg: Dcgm, log_dir: Path) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self._mode: Mode | None = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def available(self) -> Availability:
        """The running hostengine, else one we start. Neither means the diagnostic cannot run."""
        if self._mode is None:
            self._mode = choose_mode(self.cfg.discovery_timeout_s)
        if self._mode != Mode.NONE:
            return Availability(True)
        if shutil.which("dcgmi"):
            return Availability(False, "no hostengine, and nv-hostengine is not on PATH")
        return Availability(False, "dcgmi not on PATH")

    def collect(self) -> JsonDict:
        mode = self._mode
        if mode is None or mode == Mode.NONE:
            raise RuntimeError("available() did not find a hostengine")
        tests = ",".join(self.cfg.tests) or str(
            self.cfg.level
        )  # the `dcgmi diag -r` argument: the listed tests if any, else the level
        return {
            "status": Status.OK,
            "mode": mode,
            "suite": tests,
            **diag(mode, tests, self.cfg.port, self.log_dir, self.cfg.diag_timeout_s, parameters(self.cfg)),
        }

    @staticmethod
    def report(results: ByHost[JsonDict], cfg: Dcgm, name: str) -> Report:
        findings = per_host(results, cfg, name, _findings, _skipped)
        tflops_of = {
            host: gpus
            for host, data in results.items()
            if data.get("status") != Status.SKIPPED
            and (gpus := diagnostic_tflops(data))
        }
        if not tflops_of:
            return Report(findings, {})
        mid = median(
            tflops
            for gpus in tflops_of.values()
            for tflops in gpus.values()
        )
        fail_floor = (mid or 0) * cfg.diagnostic_min_frac_of_median
        warn_floor = (mid or 0) * cfg.diagnostic_warn_frac_of_median
        for host, gpus in tflops_of.items():
            for gpu, tflops in gpus.items():
                if tflops < fail_floor:
                    severity: Severity = "fail"
                    floor = fail_floor
                elif tflops < warn_floor:
                    severity = "warn"
                    floor = warn_floor
                else:
                    continue
                detail = f"GPU{gpu} {tflops} < {floor:.1f} TFLOPS (frac of median)"
                findings.setdefault(host, []).append(Finding(severity, "dcgm.tflops", detail))
        return Report(findings, {"dcgm_diagnostic_tflops": mid})


def parameters(cfg: Dcgm) -> str:
    """
    The `dcgmi diag -p` argument: cfg.parameters and diagnostic's precision.

    >>> parameters(Dcgm(parameters=["diagnostic.test_duration=30"], diagnostic_precision="fp16"))
    'diagnostic.test_duration=30;diagnostic.precision=h'
    """
    return ";".join([*cfg.parameters, f"diagnostic.precision={cfg.diagnostic_precision_letter}"])


def diagnostic_tflops(data: JsonDict) -> dict[int, float]:
    """
    Each GPU's TFLOPS from the dcgm diagnostic test's messages.

    >>> message = "GPU 1 calculated at approximately 160231.80 gigaflops"
    >>> diagnostic_tflops({"tests": {"diagnostic": {"messages": [message]}}})
    {1: 160.23}
    """
    messages = data.get("tests", {}).get("diagnostic", {}).get("messages", [])
    pattern = re.compile(r"GPU (\d+) calculated at approximately ([\d.]+) gigaflops")
    return {
        int(gpu): round(float(gflops) / 1000, 2) 
        for message 
        in messages 
            for gpu, gflops 
            in pattern.findall(message)
    }


def _skipped(data: JsonDict, cfg: Dcgm, name: str) -> Iterator[Finding]:
    """A skipped dcgm step still warns when the config used at report time says it should have run."""
    if cfg.enabled:
        yield Finding("warn", name, f"not run (dcgm={data.get('mode')})")


def _findings(data: JsonDict, cfg: Dcgm) -> Iterator[Finding]:
    found = False
    for test_name, test in data["tests"].items():
        if test["status"] in cfg.fail_on:
            severity: Severity = "fail"
        elif test["status"] in cfg.warn_on:
            severity = "warn"
        else:
            continue
        found = True
        bad = [entity for entity, status in test["entities"].items() if status == test["status"]]
        detail = f"{test_name} {test['status']} on {','.join(bad)}"
        message = f"{detail}: {test['messages'][0]}" if test["messages"] else detail
        yield Finding(severity, "dcgm", message)
    if not found and data.get("returncode"):
        yield Finding("fail", "dcgm", f"dcgmi diag exited {data['returncode']} with no failed test")
