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


def test_step_skips_when_disabled(monkeypatch, tmp_path):
    def checked_tools():
        raise AssertionError("checked tools")

    monkeypatch.setattr(dcgm, "choose_mode", checked_tools)
    result = step(dcgm.Check(Dcgm(enabled=False), tmp_path))
    assert result["status"] == "ok"
    assert result["data"] == {"status": "skipped"}


def test_available_without_dcgmi(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: None)
    found = dcgm.Check(Dcgm(enabled=True), tmp_path).available()
    assert found.available is False
    assert found.reason == "dcgmi not on PATH"


def test_available_without_hostengine(monkeypatch, tmp_path):
    monkeypatch.setattr(dcgm.shutil, "which", lambda name: "/usr/bin/dcgmi" if name == "dcgmi" else None)
    monkeypatch.setattr(dcgm, "run", lambda cmd, **kwargs: Cmd(1, "", ""))
    found = dcgm.Check(Dcgm(enabled=True), tmp_path).available()
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
    result = step(dcgm.Check(Dcgm(enabled=True), tmp_path))
    assert result["status"] == "unavailable"
    assert result["error"] == "dcgmi not on PATH"
    assert "data" not in result
