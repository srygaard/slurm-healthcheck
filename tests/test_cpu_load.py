import os
import threading
from pathlib import Path

from healthcheck import cpu_load
from healthcheck.config import CpuLoad


def fake_sysfs(root: Path, packages: dict[int, list[int]], counters: dict[int, dict[str, int]] | None = None) -> Path:
    """cpu<N>/topology/physical_package_id for each CPU, and a thermal_throttle directory with its counters."""
    for socket, cpus in packages.items():
        for cpu in cpus:
            directory = root / "devices/system/cpu" / f"cpu{cpu}"
            (directory / "topology").mkdir(parents=True)
            (directory / "topology/physical_package_id").write_text(f"{socket}\n")
            if counters and cpu in counters:
                (directory / "thermal_throttle").mkdir()
                for name, value in counters[cpu].items():
                    (directory / "thermal_throttle" / name).write_text(f"{value}\n")
    return root


def test_short_hold_keeps_throttle_deltas_but_no_clock():
    cpu_0 = min(os.sched_getaffinity(0))
    stop_event = threading.Event()
    stop_event.set()  # dcgm skipped: its future is done before the load starts
    out = cpu_load.HealthCheck(CpuLoad(enabled=True), stop_event, [cpu_0]).collect()
    assert out["status"] == "ok"
    assert out["packages"]  # throttle events are still counted
    assert all(package["full_power_mhz"] is None for package in out["packages"].values())  # too short for a clock


def test_packages_split_the_jobs_cpus_by_socket_and_burn_a_share_of_each(tmp_path):
    root = fake_sysfs(tmp_path, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})
    packages = cpu_load.CpuPackage.of_job([0, 1, 2, 3, 4, 5, 6, 7], 0.5, root)
    assert [(package.socket, package.cpus, package.busy) for package in packages] == [
        (0, [0, 1, 2, 3], [0, 2]),
        (1, [4, 5, 6, 7], [4, 6]),
    ]


def test_a_cpu_with_no_topology_counts_under_socket_0(tmp_path):
    packages = cpu_load.CpuPackage.of_job([3, 4], 1.0, tmp_path)
    assert [(package.socket, package.cpus) for package in packages] == [(0, [3, 4])]


def test_package_mhz_is_the_mean_of_its_busy_cores_only():
    package = cpu_load.CpuPackage(socket=0, cpus=[0, 1, 2], busy=[0, 1])
    assert package.busy_mhz({0: 2000.0, 1: 3000.0, 2: 200.0}) == 2500
    assert package.busy_mhz({2: 200.0}) is None


def test_throttles_sum_core_counters_and_read_the_package_counter_once(tmp_path):
    counters = {
        cpu: {
            "core_throttle_count": 1 if cpu < 2 else 0,
            "package_throttle_count": 5,  # the same package-wide value in every CPU's directory
            "core_power_limit_count": 0,
            "package_power_limit_count": 2 if cpu else 0,
        }
        for cpu in range(4)
    }
    root = fake_sysfs(tmp_path, {0: [0, 1, 2, 3]}, counters)
    (package,) = cpu_load.CpuPackage.of_job([0, 1, 2, 3], 1.0, root)
    assert package.throttles() == {"thermal": 2 + 5, "power_limit": 0 + 2}


def test_throttles_are_none_without_counters(tmp_path):
    root = fake_sysfs(tmp_path, {0: [0, 1]})
    (package,) = cpu_load.CpuPackage.of_job([0, 1], 1.0, root)
    assert package.throttles() is None


def test_missing_power_limit_files_count_as_zero(tmp_path):
    root = fake_sysfs(tmp_path, {0: [0]}, {0: {"core_throttle_count": 3, "package_throttle_count": 1}})
    (package,) = cpu_load.CpuPackage.of_job([0], 1.0, root)
    assert package.throttles() == {"thermal": 4, "power_limit": 0}


def test_the_full_power_clock_is_the_median_so_halted_readings_do_not_set_it():
    # Eight readings at 2300 MHz under full GPU power, two halted at 200 MHz, after a 3200 MHz start-up reading.
    rows = [(0, 3200, 300)] + [(5 * i, 2300, 7000) for i in range(1, 9)] + [(45, 200, 7000), (50, 200, 7000)]
    samples = [{"t_s": t, "mhz": {"0": mhz, "1": 2300}, "gpu_w": w} for t, mhz, w in rows]
    summary = cpu_load.summarize(samples, CpuLoad(min_full_power_s=15))
    assert summary["full_power_mhz"] == {"0": 2300, "1": 2300}


def test_old_samples_with_one_busy_mhz_summarize_as_package_all():
    rows = [(0, 3200, 300)] + [(5 * i, 2300, 7000) for i in range(1, 5)]
    samples = [{"t_s": t, "busy_mhz": mhz, "gpu_w": w} for t, mhz, w in rows]
    assert cpu_load.summarize(samples, CpuLoad(min_full_power_s=15))["full_power_mhz"] == {"all": 2300}


def test_burner_is_pinned_and_idle_priority():
    cpu_0 = min(os.sched_getaffinity(0))
    burner = cpu_load.CpuPackage.start_burner(cpu_0)
    try:
        assert os.sched_getaffinity(burner.pid) == {cpu_0}
        assert os.sched_getscheduler(burner.pid) == os.SCHED_IDLE
    finally:
        burner.kill()
        burner.wait()


def test_clock_sag_against_own_baseline_warns_then_fails():
    cfg = CpuLoad()  # warn under 85%, fail under 60% of the baseline

    def sag(mhz):
        package = {"throttle_delta": {"thermal": 0, "power_limit": 0}, "full_power_mhz": mhz, "baseline_mhz": 3200}
        return [(f.severity, f.check) for f in cpu_load.CpuPackage.judge("0", package, 0, cfg)]

    assert sag(3100) == []
    assert sag(2350) == [("warn", "cpu_load.sag")]  # trig0151 at 7.97 kW
    assert sag(1800) == [("fail", "cpu_load.sag")]


def test_unreadable_gpu_power_is_left_out_of_the_baseline():
    rows = [(0, 3200, 300), (5, 3200, 1000), (10, 3200, 2000)]  # ramp: the baseline
    rows += [(15 + 5 * k, 2300, 8000) for k in range(4)]  # full power
    rows += [(35 + 5 * k, 2300, None) for k in range(4)]  # full power, but NVML failed
    samples = [{"t_s": t, "mhz": {"0": mhz}, "gpu_w": w} for t, mhz, w in rows]
    summary = cpu_load.summarize(samples, CpuLoad(min_full_power_s=15))
    assert summary["baseline_mhz"] == {"0": 3200}
    assert summary["full_power_mhz"] == {"0": 2300}
