import os
import threading

from healthcheck import cpu_load
from healthcheck.config import CpuLoad

CPU = min(os.sched_getaffinity(0))  # CPU 0 may be outside the cpuset (cgroups, Slurm)


def test_skipped_when_dcgm_returns_early():
    gpus_done = threading.Event()
    gpus_done.set()  # dcgm skipped: its future is done before the load starts
    out = cpu_load.collect(CpuLoad(enabled=True), gpus_done, [CPU])
    assert out["status"] == "skipped"
    assert out["reason"].startswith("dcgm returned after")
    assert "full_power_mhz" not in out  # nothing for the report's median


def test_burner_is_pinned_and_idle_priority():
    burner = cpu_load.start_burner(CPU)
    try:
        assert os.sched_getaffinity(burner.pid) == {CPU}
        assert os.sched_getscheduler(burner.pid) == os.SCHED_IDLE
    finally:
        burner.kill()
        burner.wait()
