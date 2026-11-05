import json

import pytest

from healthcheck import dcgm
from healthcheck.check import step
from healthcheck.config import Dcgm
from healthcheck.util.cmd import Cmd, ExitCode


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
    result = step(dcgm.HealthCheck(Dcgm(enabled=False), tmp_path))
    assert result["status"] == "ok"
    assert result["data"] == {"status": "skipped"}


def test_available_without_dcgmi(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: None)
    found = dcgm.HealthCheck(Dcgm(enabled=True), tmp_path).available()
    assert found.available is False
    assert found.reason == "dcgmi not on PATH"


def test_available_without_hostengine(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: "/usr/bin/dcgmi" if name == "dcgmi" else None)
    monkeypatch.setattr(dcgm, "run", lambda cmd, **kwargs: Cmd(1, "", ""))
    found = dcgm.HealthCheck(Dcgm(enabled=True), tmp_path).available()
    assert found.available is False
    assert found.reason == "no hostengine, and nv-hostengine is not on PATH"


def test_diag_times_out(monkeypatch, tmp_path):
    seen = {}
    timeout_s = Dcgm().diag_timeout_s

    def fake_run(cmd, timeout, **kwargs):
        seen["timeout"] = timeout
        return Cmd(ExitCode.TIMEOUT, "", f"timeout after {timeout}s")

    monkeypatch.setattr(dcgm, "run", fake_run)
    with pytest.raises(dcgm.DiagError, match=f"timeout after {timeout_s}s"):
        dcgm.diag(dcgm.Mode.HOST, "memory", 5556, tmp_path, timeout_s)
    assert seen == {"timeout": timeout_s}
    assert (tmp_path / "diag.txt").read_text() == ""
    assert (tmp_path / "diag.stderr.txt").read_text() == f"timeout after {timeout_s}s"


def test_diag_writes_stdout_and_stderr(monkeypatch, tmp_path):
    result_row = {"entity_id": 0, "status": "Pass", "info": ["GPU to Host bandwidth: 55 GB/s"]}
    test = {"name": "memory", "results": [result_row]}
    stdout = json.dumps({"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [test]}]}})
    monkeypatch.setattr(dcgm, "run", lambda cmd, timeout, **kwargs: Cmd(0, stdout, "nvvs: note"))
    result = dcgm.diag(dcgm.Mode.HOST, "memory", 5556, tmp_path, 10)
    assert (tmp_path / "diag.txt").read_text() == stdout
    assert (tmp_path / "diag.stderr.txt").read_text() == "nvvs: note"
    assert result["tests"]["memory"]["status"] == "Pass"
    assert "messages" not in result["tests"]["memory"]
    assert "detail" not in result["tests"]["memory"]


def test_self_hosted_diag_does_not_use_shared_run(monkeypatch, tmp_path):
    def shared_run(*args, **kwargs):
        raise AssertionError("shared run")

    monkeypatch.setattr(dcgm, "run", shared_run)
    monkeypatch.setattr(
        dcgm, "run_hostengine", lambda cmd, env, timeout_s: Cmd(ExitCode.TIMEOUT, "", f"timeout after {timeout_s}s")
    )
    timeout_s = Dcgm().diag_timeout_s
    with pytest.raises(dcgm.DiagError, match=f"timeout after {timeout_s}s"):
        dcgm.diag(dcgm.Mode.HOST_SELF, "memory", 5556, tmp_path, timeout_s)


def test_step_unavailable_does_not_collect(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm, "choose_mode", lambda timeout_s: dcgm.Mode.NONE)
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: None)

    def diag(*args, **kwargs):
        raise AssertionError("collected")

    monkeypatch.setattr(dcgm, "diag", diag)
    result = step(dcgm.HealthCheck(Dcgm(enabled=True), tmp_path))
    assert result["status"] == "unavailable"
    assert result["error"] == "dcgmi not on PATH"
    assert "data" not in result
