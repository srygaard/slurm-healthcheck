import pynvml

from healthcheck import share, sysinfo
from healthcheck.check import Availability, step
from healthcheck.config import Scratch


def test_share_available_when_base_missing(tmp_path):
    missing = tmp_path / "nope"
    found = share.Check(missing, Scratch(enabled=True), "host").available()
    assert found == Availability(False, f"{missing} is not a directory")


def test_step_skips_disabled_share_without_probing(tmp_path):
    result = step(share.Check(tmp_path / "nope", Scratch(enabled=False), "host"))
    assert result["status"] == "ok"
    assert result["data"] == {"status": "skipped"}


def test_sysinfo_available_when_nvml_fails(monkeypatch):
    def fail():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)

    monkeypatch.setattr(sysinfo.pynvml, "nvmlInit", fail)
    found = sysinfo.Check("host", "InfiniBand").available()
    assert found == Availability(False, "NVML Shared Library Not Found")
