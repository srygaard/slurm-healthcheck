import json

import pytest

from healthcheck import dcgm, nccl, sysinfo
from healthcheck.check import Finding
from healthcheck.config import Config, Project, Scratch, gpu_gres
from launcher import report


def ok(data):
    return {"status": "ok", "elapsed_s": 1.0, "data": data}


def healthy(host):
    port = {
        "port": 1,
        "state": "ACTIVE",
        "phys_state": "LinkUp",
        "rate": "400 Gb/sec (4X NDR)",
        "rate_gbps": 400.0,
        "link_layer": "InfiniBand",
    }
    device = {
        "index": 0,
        "name": "NVIDIA H100 80GB HBM3",
        "vbios": "96.00.89.00.01",
        "ecc_enabled": True,
        "mig_enabled": False,
        "pcie_gen_max": 5,
        "pcie_width": 16,
        "nvlinks_active": 18,
    }
    counter = {
        "index": 0,
        "ecc_volatile": {"uncorrected": 0, "corrected": 0},
        "ecc_aggregate": {"uncorrected": 0, "corrected": 0},
        "remapped_rows": {"corrected": 0, "uncorrected": 0, "pending": 0, "failure": 0},
        "nvlink_errors": {"replay": 0},
        "hw_throttle": [],
        "power_limit_w": 700.0,
        "default_power_limit_w": 700.0,
    }
    sysinfo = {
        "kernel": "5.14.0",
        "os_release": "AlmaLinux 9",
        "bios": {"bios_version": "2.11.2"},
        "gpu": {
            "count": 4,
            "driver_version": "580.1",
            "cuda_version": "13.0",
            "topo_hash": "abc",
            "devices": [dict(device, index=index) for index in range(4)],
        },
        "counters": [dict(counter, index=index) for index in range(4)],
        "nic": {
            "ib": [{"name": f"mlx5_{index}", "fw_ver": "28.1", "ports": [port]} for index in range(4)],
            "net": {"ib0": {"mtu": 4092}},
            "nvidia_peermem": True,
        },
        "software": {"ofed": "MLNX_OFED-24.10"},
    }
    scratch = {"status": "ok", "path": "f0", "gbps": 2.0}
    local_tmp = {
        "paths": {
            "/tmp": {"free_gib": 3000, "write": {"gbps": 3.0}, "read": {"gbps": 5.0}},
            "/localscratch": {"free_gib": 3000, "same_device_as": "/tmp"},
        },
        "smart": {"status": "unavailable", "disks": {}},
    }
    logs = {
        "kernel": {"dmesg": {"status": "unavailable"}},
        "nccl": {"files": 4, "count": 0, "matches": []},
        "xid": {"status": "ok", "events": []},
        "counter_deltas": [],
    }
    return {
        "host": host,
        "sysinfo": ok(sysinfo),
        "scratch": ok(scratch),
        "project": ok(dict(scratch)),
        "local_tmp": ok(local_tmp),
        "dcgm": ok({"status": "ok", "tests": {"memory": {"status": "Pass", "entities": {}}}}),
        "cpu_load": ok({"status": "skipped"}),
        "logs": ok(logs),
    }


def nccl_test(kind, nodes, busbw, status="ok"):
    return {"kind": kind, "nodes": nodes, "status": status, "peak_busbw": busbw, "log": "x"}


@pytest.fixture
def run(tmp_path):
    hosts = [f"rg{index}" for index in range(1, 7)]
    (tmp_path / "nodes").mkdir()
    (tmp_path / "groups").mkdir()
    for host in hosts:
        (tmp_path / "nodes" / f"{host}.json").write_text(json.dumps(healthy(host)))
    (tmp_path / "switches.json").write_text(json.dumps({"sw1": hosts}))
    (tmp_path / "jobs.json").write_text(json.dumps([{"job_id": 1, "nodes": hosts, "state": "COMPLETED"}]))
    tests = [nccl_test("intra", [host], 350) for host in hosts]
    tests += [nccl_test("loopback", [host], 40) for host in hosts]
    tests += [nccl_test("pair", [first, second], 45) for first, second in zip(hosts, hosts[1:] + hosts[:1])]
    tests.append(nccl_test("group", hosts, 40))
    tests.append(nccl_test("alltoall", hosts, 20))
    (tmp_path / "groups" / "1.json").write_text(json.dumps({"job_id": "1", "tests": tests}))
    return tmp_path


def edit(run, host, fn):
    path = run / "nodes" / f"{host}.json"
    node = json.loads(path.read_text())
    fn(node)
    path.write_text(json.dumps(node))


def switches(run):
    by_switch = json.loads((run / "switches.json").read_text())
    return {node: switch for switch, nodes in by_switch.items() for node in nodes}


def evaluate(run, cfg=None):
    return report.evaluate(run, cfg or Config(), switches(run), gpu_gres["h100"], 4)


def statuses(run, cfg=None):
    return {host: node["status"] for host, node in evaluate(run, cfg)["nodes"].items()}


def test_all_healthy(run):
    assert set(statuses(run).values()) == {"pass"}
    _, exclude, include = report.write(run, Config(), switches(run), ["fail"], gpu_gres["h100"], 4)
    assert exclude == "# no nodes to exclude\n"
    assert include == "#SBATCH --nodelist=rg[1-6]\n"


@pytest.mark.parametrize(
    "lose",
    [
        lambda path: path.write_text(""),  # never written
        lambda path: path.unlink(),  # e.g. a --any job whose nodes were learned from sacct
    ],
    ids=["empty", "missing"],
)
def test_node_without_result_is_no_report(run, lose):
    lose(run / "nodes" / "rg3.json")
    result = evaluate(run)["nodes"]["rg3"]
    assert result["status"] == "fail"
    assert result["findings"][0]["check"] == "no_report"


def test_gpu_and_nic_checks(run):
    def bad(node):
        node["sysinfo"]["data"]["gpu"]["devices"][2]["nvlinks_active"] = 12
        port = node["sysinfo"]["data"]["nic"]["ib"][1]["ports"][0]
        port.update(rate="100 Gb/sec (2X HDR)", rate_gbps=100.0)

    edit(run, "rg2", bad)
    checks = {finding["check"] for finding in evaluate(run)["nodes"]["rg2"]["findings"]}
    assert checks == {"gpu.nvlink", "homogeneity"}


def test_gpu_capped_below_its_default_power_limit_fails(run):
    def capped(node):
        node["sysinfo"]["data"]["counters"][3]["power_limit_w"] = 200.0

    edit(run, "rg2", capped)
    result = evaluate(run)["nodes"]["rg2"]
    assert [(f["severity"], f["check"], f["detail"]) for f in result["findings"]] == [
        ("fail", "gpu.power_limit", "GPU3 power limit 200 W < default 700 W")
    ]


def test_port_slower_than_its_peers_fails_that_node(run):
    def slow(node):
        node["sysinfo"]["data"]["nic"]["ib"][0]["ports"][0].update(rate="100 Gb/sec (2X HDR)", rate_gbps=100.0)

    edit(run, "rg6", slow)
    nodes = evaluate(run)["nodes"]
    assert nodes["rg6"]["status"] == "fail"
    assert "nic.ib[*].ports[*].rate" in nodes["rg6"]["findings"][0]["detail"]
    assert nodes["rg1"]["status"] == "pass"


def test_port_slow_on_every_node_is_not_flagged(run):
    """Peers-only grading: a fleet-wide rate is the baseline, so it passes. Only a node that differs is flagged."""

    def slow(node):
        node["sysinfo"]["data"]["nic"]["ib"][0]["ports"][0].update(rate="100 Gb/sec (2X HDR)", rate_gbps=100.0)

    for host in ("rg1", "rg2", "rg3", "rg4", "rg5", "rg6"):
        edit(run, host, slow)
    assert set(statuses(run).values()) == {"pass"}


def test_homogeneity_flags_minority(run):
    edit(run, "rg4", lambda node: node["sysinfo"]["data"]["gpu"].update(driver_version="570.0"))
    nodes = evaluate(run)["nodes"]
    assert nodes["rg4"]["status"] == "fail"
    assert "gpu.driver_version" in nodes["rg4"]["findings"][0]["detail"]
    assert nodes["rg1"]["status"] == "pass"


def test_homogeneity_without_majority_flags_every_node(run):
    for host, version in (("rg3", "570.0"), ("rg4", "570.0"), ("rg5", "575.0"), ("rg6", "575.0")):
        edit(run, host, lambda node, version=version: node["sysinfo"]["data"]["gpu"].update(driver_version=version))
    nodes = evaluate(run)["nodes"]
    for host in ("rg1", "rg2", "rg3", "rg4", "rg5", "rg6"):
        assert nodes[host]["status"] == "fail"
        assert "(no majority)" in nodes[host]["findings"][0]["detail"]


def test_bios_mismatch_warns(run):
    edit(run, "rg1", lambda node: node["sysinfo"]["data"]["bios"].update(bios_version="2.11.4"))
    findings = evaluate(run)["nodes"]["rg1"]["findings"]
    assert [(finding["severity"], finding["check"]) for finding in findings] == [("warn", "homogeneity")]
    assert "bios.bios_version" in findings[0]["detail"]


def cpu_load(*packages):
    """A cpu_load record with one package per (full_power_mhz, thermal, power_limit) tuple."""
    return ok(
        {
            "status": "ok",
            "packages": {
                str(socket): {
                    "busy_cpus": "0-3",
                    "throttle_delta": {"thermal": thermal, "power_limit": power_limit},
                    "full_power_mhz": mhz,
                }
                for socket, (mhz, thermal, power_limit) in enumerate(packages)
            },
            "samples": [],
        }
    )


def test_cpu_load_throttle_events_and_slow_package_warn_per_package(run):
    for host in ("rg1", "rg2", "rg3", "rg4", "rg5"):
        edit(run, host, lambda node: node.update(cpu_load=cpu_load((2300, 0, 0), (2300, 0, 0))))
    edit(run, "rg6", lambda node: node.update(cpu_load=cpu_load((2300, 0, 0), (1500, 0, 3))))
    report_ = evaluate(run)
    assert report_["nodes"]["rg1"]["status"] == "pass"
    findings = report_["nodes"]["rg6"]["findings"]
    assert [(finding["severity"], finding["check"]) for finding in findings] == [
        ("warn", "cpu_load.throttle"),
        ("warn", "cpu_load.median"),
    ]
    assert findings[0]["detail"] == "package 1: 0 thermal and 3 power-limit throttle events"
    assert findings[1]["detail"].startswith("package 1: busy-core clock 1500 < ")
    assert report_["nodes"]["rg6"]["status"] == "warn"
    assert report_["medians"]["cpu_load_busy_mhz"] == 2300


def test_a_run_where_every_package_drops_together_is_not_flagged(run):
    for host in ("rg1", "rg2", "rg3", "rg4", "rg5", "rg6"):
        edit(run, host, lambda node: node.update(cpu_load=cpu_load((1700, 0, 0), (1750, 0, 0))))
    assert set(statuses(run).values()) == {"pass"}


def test_a_record_from_before_packages_still_grades(run):
    old = {"status": "ok", "full_power_mhz": 2300, "low_share": 0.9, "throttle_delta": 0, "samples": []}
    for host in ("rg1", "rg2", "rg3", "rg4", "rg5"):
        edit(run, host, lambda node: node.update(cpu_load=ok(dict(old))))
    edit(run, "rg6", lambda node: node.update(cpu_load=ok({**old, "full_power_mhz": 1500, "throttle_delta": 2})))
    findings = evaluate(run)["nodes"]["rg6"]["findings"]
    assert [finding["check"] for finding in findings] == ["cpu_load.throttle", "cpu_load.median"]
    assert findings[0]["detail"] == "2 thermal and 0 power-limit throttle events"
    assert statuses(run)["rg1"] == "pass"


def test_unreadable_throttle_counters_are_not_findings(run):
    package = {"busy_cpus": "0-3", "throttle_delta": None, "full_power_mhz": 2300}
    edit(run, "rg1", lambda node: node.update(cpu_load=ok({"status": "ok", "packages": {"0": package}, "samples": []})))
    assert statuses(run)["rg1"] == "pass"


def test_historical_corrected_ecc_does_not_warn(run):
    """Lifetime corrected counters that did not rise during the run are not findings; logs grades increases."""

    def corrected(node):
        node["sysinfo"]["data"]["counters"][1]["ecc_aggregate"]["corrected"] = 177903
        node["sysinfo"]["data"]["counters"][1]["remapped_rows"]["corrected"] = 2

    edit(run, "rg2", corrected)
    result = evaluate(run)["nodes"]["rg2"]
    assert not [finding for finding in result["findings"] if finding["check"] in ("gpu.ecc", "gpu.remap")]


def test_known_kernel_lines_warn_and_other_lines_fail(run):
    dmesg = {
        "status": "ok",
        "matches": [
            "NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0",
            "mlx5_core 0000:d8:00.1: mlx5_query_module_id:314: query_mcia_reg failed: status: 0x3",
            "NVRM: Xid (PCI:0000:19:00): 79, GPU has fallen off the bus",
        ],
    }
    edit(run, "rg1", lambda node: node["logs"]["data"].update(kernel={"dmesg": dict(dmesg)}))
    findings = evaluate(run)["nodes"]["rg1"]["findings"]
    assert [(finding["severity"], finding["check"]) for finding in findings] == [
        ("warn", "logs.dmesg"),
        ("warn", "logs.dmesg"),
        ("fail", "logs.dmesg"),
    ]
    assert "inband data: 0" in findings[0]["detail"]
    assert "query_mcia_reg failed" in findings[1]["detail"]
    assert "Xid" in findings[2]["detail"]
    assert evaluate(run)["nodes"]["rg1"]["status"] == "fail"


def test_cpu_load_skipped_or_absent_passes(run):
    edit(run, "rg1", lambda node: node.pop("cpu_load"))  # a run from before the step existed
    assert set(statuses(run).values()) == {"pass"}
    assert "cpu_load_busy_mhz" not in evaluate(run)["medians"]


def test_dcgm_tflops_relative_to_median(run):
    def diagnostic(tflops):
        stored = {str(gpu): value for gpu, value in enumerate(tflops)}
        return lambda node: node["dcgm"]["data"]["tests"].update(
            diagnostic={"status": "Pass", "entities": {}, "tflops": stored}
        )

    for host in ("rg1", "rg2", "rg3", "rg4", "rg5"):
        edit(run, host, diagnostic([160.23, 164.94, 160.23, 164.94]))
    edit(run, "rg6", diagnostic([160.23, 150.0, 140.0, 160.23]))
    result = evaluate(run)
    assert [(finding["severity"], finding["detail"]) for finding in result["nodes"]["rg6"]["findings"]] == [
        ("warn", "GPU1 150.0 < 152.2 TFLOPS (frac of median)"),
        ("fail", "GPU2 140.0 < 144.2 TFLOPS (frac of median)"),
    ]
    assert result["nodes"]["rg6"]["status"] == "fail"
    assert {result["nodes"][host]["status"] for host in ("rg1", "rg2", "rg3", "rg4", "rg5")} == {"pass"}
    assert result["medians"]["dcgm_diagnostic_tflops"] == 160.23


def test_scratch_relative_to_median(run):
    edit(run, "rg5", lambda node: node["scratch"]["data"].update(gbps=0.8))  # above min_GBps 0.2
    assert statuses(run)["rg5"] == "warn"  # severity: warn
    assert statuses(run, Config(scratch=Scratch(severity="fail")))["rg5"] == "fail"


def test_project_read_has_its_own_thresholds(run):
    edit(run, "rg5", lambda node: node["project"]["data"].update(gbps=0.8))  # above min_GBps, below half the median
    result = evaluate(run)
    assert [finding["check"] for finding in result["nodes"]["rg5"]["findings"]] == ["project.read"]
    assert result["medians"]["project_read_GBps"] == 2.0
    assert statuses(run, Config(project=Project(severity="fail")))["rg5"] == "fail"
    assert statuses(run, Config(scratch=Scratch(severity="fail")))["rg5"] == "warn"


def test_disabled_reads_are_skipped(run):
    for host in ("rg1", "rg2"):
        edit(run, host, lambda node: node.update(scratch=ok({"status": "skipped"}), project=ok({"status": "skipped"})))
    result = evaluate(run)
    assert result["nodes"]["rg1"]["findings"] == []
    assert result["medians"]["scratch_read_GBps"] == 2.0  # from the nodes that ran it
    assert result["medians"]["project_read_GBps"] == 2.0


def test_disabled_reads_have_no_median(run):
    for host in ("rg1", "rg2", "rg3", "rg4", "rg5", "rg6"):
        edit(run, host, lambda node: node.update(project=ok({"status": "skipped"})))
    medians = evaluate(run)["medians"]
    assert "project_read_GBps" not in medians
    assert medians["scratch_read_GBps"] == 2.0


def test_local_tmp_same_device_checks_free_space_only(run):
    edit(run, "rg3", lambda node: node["local_tmp"]["data"]["paths"]["/localscratch"].update(free_gib=10))
    findings = evaluate(run)["nodes"]["rg3"]["findings"]
    assert [finding["check"] for finding in findings] == ["local_tmp.free"]


def test_bad_pair_node_fails_neighbours_warn(run):
    group = json.loads((run / "groups" / "1.json").read_text())
    for test in group["tests"]:
        if test["kind"] == "pair" and "rg3" in test["nodes"]:
            test["peak_busbw"] = 5
    (run / "groups" / "1.json").write_text(json.dumps(group))
    by_host = statuses(run)
    assert by_host["rg3"] == "fail"
    assert by_host["rg2"] == by_host["rg4"] == "warn"
    assert by_host["rg1"] == "pass"
    _, exclude, include = report.write(run, Config(), switches(run), ["fail"], gpu_gres["h100"], 4)
    assert exclude == "#SBATCH --exclude=rg3\n"
    assert include == "#SBATCH --nodelist=rg[1-2,4-6]\n"
    _, exclude, include = report.write(run, Config(), switches(run), ["fail", "warn"], gpu_gres["h100"], 4)
    assert exclude == "#SBATCH --exclude=rg[2-4]\n"
    assert include == "#SBATCH --nodelist=rg[1,5-6]\n"
    assert (run / "include.sbatch").read_text() == include


def test_intra_below_fail_fraction_fails_and_the_warn_band_warns(run):
    """350 GB/s median: 300 is under the 0.9 line (315) and over the 120 GB/s floor; 320 is in the 0.95 warn band."""

    def set_intra(busbw):
        def change(test, busbw=busbw):
            if test["kind"] == "intra" and test["nodes"] == ["rg3"]:
                test["peak_busbw"] = busbw

        set_tests(run, lambda test: True, change)

    set_intra(300)
    findings = evaluate(run)["nodes"]["rg3"]["findings"]
    assert findings[0] == {"severity": "fail", "check": "nccl.intra", "detail": "busbw 300 < 315.0"}
    set_intra(320)
    findings = evaluate(run)["nodes"]["rg3"]["findings"]
    assert findings[0] == {"severity": "warn", "check": "nccl.intra", "detail": "busbw 320 < 332.5"}
    set_intra(None)  # an ok result with no bandwidth is graded as 0, and the message says 0
    findings = evaluate(run)["nodes"]["rg3"]["findings"]
    assert findings[0]["detail"] == "busbw 0 < 315.0"


def test_pair_below_fail_fraction_fails_that_node(run):
    """86% of a 45 GB/s median is under the 0.9 fail line and over the 20 GB/s floor."""
    group = json.loads((run / "groups" / "1.json").read_text())
    for test in group["tests"]:
        if test["kind"] == "pair" and "rg3" in test["nodes"]:
            test["peak_busbw"] = 38.7
    (run / "groups" / "1.json").write_text(json.dumps(group))
    by_host = statuses(run)
    assert by_host["rg3"] == "fail"
    assert by_host["rg2"] == by_host["rg4"] == "warn"
    assert by_host["rg1"] == "pass"
    detail = evaluate(run)["nodes"]["rg3"]["findings"][0]["detail"]
    assert detail.startswith("2/2 slow pairs (< 40.5)")


def test_pair_between_warn_and_fail_warns_both_nodes(run):
    """92% of a 45 GB/s median is under the 0.95 warn line and over the 0.9 fail line."""
    group = json.loads((run / "groups" / "1.json").read_text())
    for test in group["tests"]:
        if test["kind"] == "pair" and "rg3" in test["nodes"]:
            test["peak_busbw"] = 41.4
    (run / "groups" / "1.json").write_text(json.dumps(group))
    by_host = statuses(run)
    assert by_host["rg3"] == by_host["rg2"] == by_host["rg4"] == "warn"
    assert by_host["rg1"] == "pass"
    findings = evaluate(run)["nodes"]["rg3"]["findings"]
    assert findings[0]["check"] == "nccl.pair"
    assert findings[0]["detail"].startswith("2/2 slow pairs (< 42.8)")


@pytest.mark.parametrize(
    "change",
    [
        lambda node: node["logs"]["data"]["xid"]["events"].append({"gpu": 1, "xid": 79, "time": "t"}),
        lambda node: node["dcgm"]["data"]["tests"]["memory"].update(status="Fail", entities={"2": "Fail"}),
        lambda node: node["dcgm"]["data"].update(returncode=226),  # nonzero exit without a failed test
    ],
    ids=["xid", "dcgm_test_failed", "dcgm_nonzero_exit"],
)
def test_node_fails(run, change):
    edit(run, "rg1", change)
    assert statuses(run)["rg1"] == "fail"


def test_resolve_spread_over_dict():
    assert sysinfo.resolve({"net": {"a": {"mtu": 1}, "b": {"mtu": 2}}}, "net[*].mtu") == (1, 2)
    assert sysinfo.resolve({}, "gpu.driver_version") is None


def test_dcgm_parse_v4_layout():
    out = json.dumps(
        {
            "DCGM Diagnostic": {
                "test_categories": [
                    {
                        "category": "Hardware",
                        "tests": [
                            {
                                "name": "memory",
                                "results": [
                                    {"entity_id": 0, "status": "Pass"},
                                    {"entity_id": 1, "status": "Fail", "warnings": [{"warning": "DBE detected"}]},
                                ],
                            }
                        ],
                    },
                    {"category": "Deployment", "tests": [{"name": "software", "test_summary": {"status": "Pass"}}]},
                ]
            }
        }
    )
    tests = dcgm.parse("banner\n" + out)["tests"]
    assert tests["memory"]["status"] == "Fail"
    assert tests["memory"]["detail"] == "DBE detected"
    assert "messages" not in tests["memory"]
    assert tests["software"]["status"] == "Pass"


def test_ecc_increase_reported_once(run):
    delta = {
        "index": 0,
        "hw_throttle": [],
        "increased": {"ecc_volatile.uncorrected": 1, "ecc_aggregate.uncorrected": 1},
    }
    edit(run, "rg2", lambda node: node["logs"]["data"]["counter_deltas"].append(delta))
    findings = evaluate(run)["nodes"]["rg2"]["findings"]
    assert [finding["detail"] for finding in findings] == ["GPU0 ecc_volatile.uncorrected +1"]


def test_runtimes(run):
    result = evaluate(run)
    assert result["jobs"] == [
        {"job_id": 1, "label": None, "nodes": 6, "state": "COMPLETED", "elapsed_s": None, "nccl_elapsed_s": None}
    ]
    assert result["nodes"]["rg1"]["elapsed_s"] == dict.fromkeys(report.STEPS, 1.0)


def set_tests(run, keep, change):
    """Rewrite the NCCL tests: drop those keep rejects, then apply change to each remaining one."""
    group = json.loads((run / "groups" / "1.json").read_text())
    group["tests"] = [test for test in group["tests"] if keep(test)]
    for test in group["tests"]:
        change(test)
    (run / "groups" / "1.json").write_text(json.dumps(group))


def time_out(test):
    test["status"] = "timeout"
    test["peak_busbw"] = None


def test_two_node_pair_timeout_excludes_both(run):
    set_tests(
        run,
        lambda test: test["kind"] in ("intra", "loopback") or test["nodes"] == ["rg1", "rg2"],
        lambda test: time_out(test) if test["kind"] == "pair" else None,
    )
    by_host = statuses(run)
    assert by_host["rg1"] == by_host["rg2"] == "fail"
    assert by_host["rg3"] == "pass"
    assert "timeout 0" in evaluate(run)["nodes"]["rg1"]["findings"][0]["detail"]


def test_group_timeout_blames_the_pair_culprit_only(run):
    def hang_rg3_and_group(test):
        if test["kind"] == "group" or (test["kind"] == "pair" and "rg3" in test["nodes"]):
            time_out(test)

    set_tests(run, lambda test: True, hang_rg3_and_group)
    by_host = statuses(run)
    assert by_host["rg3"] == "fail"
    assert by_host["rg1"] == by_host["rg2"] == by_host["rg4"] == "warn"


def test_slow_alltoall_warns_every_node(run):
    set_tests(run, lambda test: True, lambda test: test.update(peak_busbw=0.5) if test["kind"] == "alltoall" else None)
    assert set(statuses(run).values()) == {"warn"}  # below nccl.alltoall.min_busbw_GBps 1.0
    assert evaluate(run)["medians"]["nccl_alltoall_busbw_GBps"] == 0.5


def test_group_timeout_without_culprit_excludes_the_group(run):
    set_tests(run, lambda test: True, lambda test: time_out(test) if test["kind"] == "group" else None)
    assert set(statuses(run).values()) == {"fail"}
    assert "busbw 0 " in evaluate(run)["nodes"]["rg1"]["findings"][0]["detail"]
    _, exclude, include = report.write(run, Config(), switches(run), ["fail"], gpu_gres["h100"], 4)
    assert exclude == "#SBATCH --exclude=rg[1-6]\n"
    assert include == "# no nodes to include\n"


def test_alltoall_timeout_warns_every_node(run):
    set_tests(run, lambda test: True, lambda test: time_out(test) if test["kind"] == "alltoall" else None)
    assert set(statuses(run).values()) == {"warn"}
    _, exclude, _ = report.write(run, Config(), switches(run), ["fail"], gpu_gres["h100"], 4)
    assert exclude == "# no nodes to exclude\n"


def test_nccl_without_mpi_warns_instead_of_failing():
    result = nccl.HealthCheck.report(
        [{"job_id": 1, "nodes": ["cn-l001"], "mpi": "none", "tests": [], "elapsed_s": 0}],
        Config().nccl,
        "nccl",
    )
    assert result.findings["cn-l001"] == [Finding("warn", "nccl", "skipped: no MPI plugin")]
    assert result.medians == {}


def test_unavailable_step_warns_and_does_not_fail(run):
    edit(
        run,
        "rg2",
        lambda node: node.update(sysinfo={"status": "unavailable", "error": "Driver Not Loaded", "elapsed_s": 0.1}),
    )
    result = evaluate(run)["nodes"]["rg2"]
    assert result["status"] == "warn"
    assert result["findings"] == [{"severity": "warn", "check": "sysinfo", "detail": "unavailable: Driver Not Loaded"}]


def test_step_error_fails(run):
    edit(run, "rg2", lambda node: node.update(dcgm={"status": "error", "error": "DiagError: boom", "elapsed_s": 1}))
    result = evaluate(run)["nodes"]["rg2"]
    assert result["status"] == "fail"
    assert result["findings"][0]["detail"] == "error: DiagError: boom"


def test_nccl_unavailable_warns():
    result = nccl.HealthCheck.report(
        [
            {
                "job_id": 1,
                "nodes": ["cn-l001"],
                "status": "unavailable",
                "error": "missing all_reduce_perf in /build",
                "tests": [],
                "elapsed_s": 0,
            }
        ],
        Config().nccl,
        "nccl",
    )
    assert result.findings["cn-l001"] == [Finding("warn", "nccl", "unavailable: missing all_reduce_perf in /build")]
    assert result.medians == {}

