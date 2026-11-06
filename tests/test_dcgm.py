import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthcheck import dcgm
from healthcheck.check import step
from healthcheck.config import Dcgm, GpuLimits
from healthcheck.util.cmd import Cmd, ExitCode

SXM = dict(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18)


def limits(timeout_s: float = 600):
    """A limits loader for HealthCheck: what node.py builds from the job template."""
    return lambda: GpuLimits(**SXM, dcgm_timeout_s=timeout_s)


def test_choose_mode(monkeypatch):
    binaries: set[str] = set()
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: f"/usr/bin/{name}" if name in binaries else None)
    state = {"ok": False}

    timeout_s = Dcgm().discovery_timeout_s

    def fake_run(cmd, **kwargs):
        assert cmd == ["dcgmi", "discovery", "-l"]
        assert kwargs["timeout"] == timeout_s
        return Cmd(0 if state["ok"] else 1, "", "")

    monkeypatch.setattr(dcgm, "run", fake_run)
    assert dcgm.choose_mode(timeout_s) == dcgm.Mode.NONE  # no dcgmi
    binaries.add("dcgmi")
    assert dcgm.choose_mode(timeout_s) == dcgm.Mode.NONE  # discovery fails, and we cannot start a hostengine
    binaries.add("nv-hostengine")
    assert dcgm.choose_mode(timeout_s) == dcgm.Mode.HOST_SELF
    state["ok"] = True
    assert dcgm.choose_mode(timeout_s) == dcgm.Mode.HOST


def test_parse_keeps_every_gpus_tflops_and_only_the_failure_line():
    # Three info lines per GPU. They are not stored; the TFLOPS are.
    results = [
        {
            "entity_id": gpu,
            "status": "Fail" if gpu == 7 else "Pass",
            "info": [
                f"Allocated space for GPU {gpu}",
                "Running with precisions: FP16 1",
                f"GPU {gpu} calculated at approximately 1294968.00 gigaflops during this test",
            ],
            "warnings": [{"warning": "slow"}] if gpu == 7 else [],
            "errors": [{"error": "mismatch"}] if gpu == 7 else [],
        }
        for gpu in range(8)
    ]
    test = {"name": "diagnostic", "results": results}
    report = {"DCGM Diagnostic": {"test_categories": [{"category": "Stress", "tests": [test]}]}}
    parsed = dcgm.parse("banner\n" + json.dumps(report))["tests"]["diagnostic"]
    assert "messages" not in parsed
    assert parsed["detail"] == "mismatch"
    assert sorted(int(gpu) for gpu in parsed["tflops"]) == list(range(8))
    assert sorted(dcgm.diagnostic_tflops({"tests": {"diagnostic": parsed}})) == list(range(8))


def test_step_skips_when_disabled(monkeypatch, tmp_path):
    def checked_tools():
        raise AssertionError("checked tools")

    monkeypatch.setattr(dcgm, "choose_mode", checked_tools)
    result = step(dcgm.HealthCheck(Dcgm(enabled=False), tmp_path, limits()))
    assert result["status"] == "ok"
    assert result["data"] == {"status": "skipped"}


def test_available_without_dcgmi(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: None)
    found = dcgm.HealthCheck(Dcgm(enabled=True), tmp_path, limits()).available()
    assert found.available is False
    assert found.reason == "dcgmi not on PATH"


def test_available_without_hostengine(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: "/usr/bin/dcgmi" if name == "dcgmi" else None)
    monkeypatch.setattr(dcgm, "run", lambda cmd, **kwargs: Cmd(1, "", ""))
    found = dcgm.HealthCheck(Dcgm(enabled=True), tmp_path, limits()).available()
    assert found.available is False
    assert found.reason == "no hostengine, and nv-hostengine is not on PATH"


def test_diag_times_out(monkeypatch, tmp_path):
    seen = {}
    timeout_s = 600

    def fake_run(cmd, env, timeout_s):
        seen["timeout"] = timeout_s
        return Cmd(ExitCode.TIMEOUT, "", f"timeout after {timeout_s}s")

    monkeypatch.setattr(dcgm, "run_group", fake_run)
    with pytest.raises(dcgm.DiagError, match=f"timeout after {timeout_s}s"):
        dcgm.diag(dcgm.Mode.HOST, "memory", 5556, tmp_path, timeout_s)
    assert seen == {"timeout": timeout_s}
    assert (tmp_path / "diag.txt").read_text() == ""
    assert (tmp_path / "diag.stderr.txt").read_text() == f"timeout after {timeout_s}s"


def test_diag_writes_stdout_and_stderr(monkeypatch, tmp_path):
    result_row = {"entity_id": 0, "status": "Pass", "info": ["GPU to Host bandwidth: 55 GB/s"]}
    test = {"name": "memory", "results": [result_row]}
    stdout = json.dumps({"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [test]}]}})
    monkeypatch.setattr(dcgm, "run_group", lambda cmd, env, timeout_s: Cmd(0, stdout, "nvvs: note"))
    result = dcgm.diag(dcgm.Mode.HOST, "memory", 5556, tmp_path, 10)
    assert (tmp_path / "diag.txt").read_text() == stdout
    assert (tmp_path / "diag.stderr.txt").read_text() == "nvvs: note"
    assert result["tests"]["memory"]["status"] == "Pass"
    assert "messages" not in result["tests"]["memory"]
    assert "detail" not in result["tests"]["memory"]


def test_run_group_timeout_kills_what_the_command_started(tmp_path):
    pid_file = tmp_path / "child.pid"
    started = time.monotonic()
    out = dcgm.run_group(["bash", "-c", f"sleep 30 & echo $! > {pid_file}; wait"], dict(os.environ), 0.5)
    assert out.returncode == ExitCode.TIMEOUT and time.monotonic() - started < 10
    time.sleep(0.2)
    assert not Path(f"/proc/{pid_file.read_text().strip()}").exists()  # the background sleep died with the group


def test_step_unavailable_does_not_collect(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm, "choose_mode", lambda timeout_s: dcgm.Mode.NONE)
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: None)

    def diag(*args, **kwargs):
        raise AssertionError("collected")

    monkeypatch.setattr(dcgm, "diag", diag)
    result = step(dcgm.HealthCheck(Dcgm(enabled=True), tmp_path, limits()))
    assert result["status"] == "unavailable"
    assert result["error"] == "dcgmi not on PATH"
    assert "data" not in result


@pytest.fixture
def collect(monkeypatch, tmp_path):
    """collect() on 4 GPUs, the given ones capped at the GpuLimits default 0.95; diag records its timeout."""
    monkeypatch.delenv("HC_DCGM_SKIP_TESTS", raising=False)
    timeouts = []

    def diag(mode, suite, port, log_dir, timeout_s, *args):
        timeouts.append(timeout_s)
        return {}

    monkeypatch.setattr(dcgm, "diag", diag)

    def run(capped):
        def power_limits(i):
            return SimpleNamespace(power_capped=lambda frac: frac == 0.95 and i in capped)

        states = [SimpleNamespace(index=i, power_limits=power_limits(i)) for i in range(4)]
        monkeypatch.setattr(dcgm.gpu, "states", lambda: states)
        check = dcgm.HealthCheck(Dcgm(), tmp_path, limits(900))
        check._mode = dcgm.Mode.HOST
        return check.collect() | {"timeout_s": timeouts[-1]}

    return run


def test_memory_and_pcie_are_dropped_only_on_power_capped_gpus(collect):
    healthy = collect([])
    assert {"memory", "pcie"} <= set(healthy["suite"].split(",")) and "power_capped_gpus" not in healthy
    capped = collect([0, 1])
    assert not {"memory", "pcie"} & set(capped["suite"].split(","))
    assert capped["skipped_for_power_cap"] == ["memory", "pcie"] and capped["power_capped_gpus"] == [0, 1]
    assert healthy["timeout_s"] == capped["timeout_s"] == 900  # the GPU type's timeout, not a global one


def test_cap_is_recorded_when_the_template_already_skips_the_tests(collect, monkeypatch):
    monkeypatch.setenv("HC_DCGM_SKIP_TESTS", "memory,pcie")
    capped = collect([2])
    assert capped["power_capped_gpus"] == [2] and "skipped_for_power_cap" not in capped


def test_bad_job_template_fails_only_the_dcgm_step(monkeypatch, tmp_path):
    def broken():
        raise ValueError("no job template for ?")

    monkeypatch.setattr(dcgm, "choose_mode", lambda timeout_s: dcgm.Mode.HOST)
    check = dcgm.HealthCheck(Dcgm(enabled=True), tmp_path, broken)  # constructing it must not raise
    result = step(check)
    assert result["status"] == "error" and "no job template" in result["error"]


def test_misspelt_skip_name_is_rejected(monkeypatch):
    monkeypatch.setenv("HC_DCGM_SKIP_TESTS", "PCIe")
    with pytest.raises(ValueError, match="PCIe, not in dcgm.tests"):
        Dcgm().suite
