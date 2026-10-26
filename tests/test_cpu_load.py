import os
import threading

from healthcheck import cpu_load
from healthcheck.config import CpuLoad


def test_skipped_when_dcgm_returns_early():
    gpus_done = threading.Event()
    gpus_done.set()  # dcgm skipped: its future is done before the load starts
    out = cpu_load.collect(CpuLoad(enabled=True), gpus_done, [0])
    assert out["status"] == "skipped"
    assert out["reason"].startswith("dcgm returned after")
    assert "last_mhz" not in out  # nothing for the report's median


def test_burner_is_pinned_and_idle_priority():
    burner = cpu_load.start_burner(0)
    try:
        assert os.sched_getaffinity(burner.pid) == {0}
        assert os.sched_getscheduler(burner.pid) == os.SCHED_IDLE
    finally:
        burner.kill()
        burner.wait()
