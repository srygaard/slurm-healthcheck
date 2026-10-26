from healthcheck.sysinfo import cpu_power


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n")


def test_collect(tmp_path):
    cpus = tmp_path / "devices" / "system" / "cpu"
    for cpu, max_khz in ((0, 4100000), (1, 4100000), (2, 3000000)):
        write(cpus / f"cpu{cpu}" / "cpufreq" / "scaling_governor", "performance")
        write(cpus / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq", max_khz)
    write(cpus / "intel_pstate" / "no_turbo", 0)
    zone = tmp_path / "class" / "powercap" / "intel-rapl:0"
    write(zone / "name", "package-0")
    write(zone / "enabled", 1)
    write(zone / "constraint_0_name", "long_term")
    write(zone / "constraint_0_power_limit_uw", 225000000)
    write(zone / "constraint_1_name", "short_term")

    out = cpu_power.collect(tmp_path)
    assert out["governor"] == ["performance"]
    assert out["max_mhz"] == [3000, 4100]  # one CPU capped lower shows as a second value
    assert out["epp"] == []
    assert out["no_turbo"] == "0"
    assert out["boost"] is None
    assert out["rapl"] == [
        {
            "zone": "intel-rapl:0",
            "name": "package-0",
            "enabled": "1",
            "limits_w": {"long_term": 225.0, "short_term": None},
        }
    ]


def test_collect_without_cpufreq(tmp_path):
    out = cpu_power.collect(tmp_path)
    assert out["governor"] == [] and out["rapl"] == [] and out["intel_pstate"] is None
