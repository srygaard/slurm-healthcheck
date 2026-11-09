import os
import threading
from pathlib import Path

from healthcheck import cpu_load
from healthcheck.config import CpuLoad


def fake_sysfs(root: Path, packages: dict[int, list[int]]) -> Path:
    """cpu<N>/topology/physical_package_id for each CPU."""
    for socket, cpus in packages.items():
        for cpu in cpus:
            directory = root / "devices/system/cpu" / f"cpu{cpu}/topology"
            directory.mkdir(parents=True)
            (directory / "physical_package_id").write_text(f"{socket}\n")
    return root


def fake_counter(monkeypatch, *reads: str | None) -> list[Path]:
    """Stand in for sysfs: each read_sys() call returns the next of reads. The paths read are collected."""
    paths: list[Path] = []
    values = iter(reads)

    def read_sys(path: Path) -> str | None:
        paths.append(path)
        return next(values)

    monkeypatch.setattr(cpu_load, "read_sys", read_sys)
    return paths


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


def test_thermal_throttles_read_only_the_package_counter_of_the_first_cpu(tmp_path):
    for cpu in (48, 49):
        directory = tmp_path / f"cpu{cpu}/thermal_throttle"
        directory.mkdir(parents=True)
        (directory / "core_throttle_count").write_text("1\n")  # per core: not counted
        (directory / "package_power_limit_count").write_text("2\n")  # not counted
        (directory / "package_throttle_count").write_text(f"{cpu}\n")  # the same in every CPU's directory, but not here
    assert cpu_load.CpuPackage(socket=1, cpus=[48, 49], busy=[]).thermal_throttles(tmp_path) == 48


def test_thermal_throttles_read_sysfs_by_default(monkeypatch):
    paths = fake_counter(monkeypatch, "5")
    assert cpu_load.CpuPackage(socket=1, cpus=[48, 49], busy=[]).thermal_throttles() == 5
    assert paths == [Path("/sys/devices/system/cpu/cpu48/thermal_throttle/package_throttle_count")]


def test_thermal_throttles_are_none_without_the_counter(monkeypatch):
    fake_counter(monkeypatch, None, None, None)
    package = cpu_load.CpuPackage(socket=0, cpus=[0], busy=[])  # no busy CPUs, so no burners start
    assert package.thermal_throttles() is None
    with package.burning():
        pass
    assert package.burn_throttles is None


def test_thermal_throttles_count_only_the_episodes_during_the_burn(monkeypatch):
    fake_counter(monkeypatch, "10", "12", "12", None)  # before and after one burn, then a burn that lost the counter
    package = cpu_load.CpuPackage(socket=0, cpus=[0], busy=[])
    with package.burning():
        pass
    assert package.burn_throttles == 2
    with package.burning():
        pass
    assert package.burn_throttles is None  # the last burn's episodes are dropped


def test_the_full_power_clock_is_the_median_so_halted_readings_do_not_set_it():
    # Eight readings at 2300 MHz under full GPU power, two halted at 200 MHz, after a 3200 MHz start-up reading.
    rows = [(0, 3200, 300)] + [(5 * i, 2300, 7000) for i in range(1, 9)] + [(45, 200, 7000), (50, 200, 7000)]
    samples = [{"t_s": t, "mhz": {"0": mhz, "1": 2300}, "gpu_w": w} for t, mhz, w in rows]
    summary = cpu_load.Summary.from_samples(samples, CpuLoad(min_full_power_s=15))
    assert [summary.of(socket).full_power_mhz for socket in ("0", "1")] == [2300, 2300]


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
    def sag(mhz):
        record = cpu_load.PackageRecord("0", 0, mhz, 3200)
        return [(f.severity, f.check) for f in record.findings(0, fail_frac=0.6, warn_frac=0.85)]

    assert sag(3100) == []
    assert sag(2350) == [("warn", "cpu_load.sag")]  # trig0151 at 7.97 kW
    assert sag(1800) == [("fail", "cpu_load.sag")]


def test_a_package_record_reads_back_as_written():
    record = cpu_load.PackageRecord("1", 3, 2300, 3200, "48-53,55-60")
    data = {"packages": {record.socket: record.to_json()}}  # no samples, so the clocks are read as written
    assert cpu_load.PackageRecord.of_node(data, CpuLoad()) == [record]


def test_an_older_record_with_a_throttle_delta_still_reads():
    data = {"packages": {"0": {"throttle_delta": {"thermal": 1, "power_limit": 0}, "full_power_mhz": 2300}}}
    (record,) = cpu_load.PackageRecord.of_node(data, CpuLoad())
    assert record.thermal_throttles == 1


def test_time_at_full_power_comes_from_the_runs_own_sampling_interval():
    # Sampled every second, though interval_s is 5: 20 readings at full power are 20 s, not 100 s
    rows = [(0, 3200, 300), (1, 3200, 1000), (2, 3200, 2000)] + [(3 + k, 2300, 8000) for k in range(20)]
    samples = [{"t_s": t, "mhz": {"0": mhz}, "gpu_w": w} for t, mhz, w in rows]
    assert CpuLoad().interval_s == 5
    assert cpu_load.Summary.from_samples(samples, CpuLoad(min_full_power_s=20)).of("0").full_power_mhz == 2300
    assert cpu_load.Summary.from_samples(samples, CpuLoad(min_full_power_s=30)).of("0").full_power_mhz is None


def test_unreadable_gpu_power_is_left_out_of_the_baseline():
    rows = [(0, 3200, 300), (5, 3200, 1000), (10, 3200, 2000)]  # ramp: the baseline
    rows += [(15 + 5 * k, 2300, 8000) for k in range(4)]  # full power
    rows += [(35 + 5 * k, 2300, None) for k in range(4)]  # full power, but NVML failed
    samples = [{"t_s": t, "mhz": {"0": mhz}, "gpu_w": w} for t, mhz, w in rows]
    summary = cpu_load.Summary.from_samples(samples, CpuLoad(min_full_power_s=15))
    assert summary.of("0") == cpu_load.Clocks(full_power_mhz=2300, baseline_mhz=3200)
