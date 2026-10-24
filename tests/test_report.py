import json

import pytest

from healthcheck import dcgm
from healthcheck.config import Config, Scratch, ScratchRead
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
        "remapped_rows": {"corrected": 0, "uncorrected": 0, "pending": 0, "failure": 0},
        "nvlink_errors": {"replay": 0},
        "hw_throttle": [],
    }
    sysinfo = {
        "kernel": "5.14.0",
        "os_release": "AlmaLinux 9",
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
        "local_tmp": ok(local_tmp),
        "dcgm": ok({"status": "ok", "tests": {"memory": {"status": "Pass", "entities": {}, "messages": []}}}),
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
    return report.evaluate(run, cfg or Config(), switches(run))


def statuses(run, cfg=None):
    return {host: node["status"] for host, node in evaluate(run, cfg)["nodes"].items()}


def test_all_healthy(run):
    assert set(statuses(run).values()) == {"pass"}
    _, exclude, include = report.write(run, Config(), switches(run), ["fail"])
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
        node["sysinfo"]["data"]["nic"]["ib"][1]["ports"][0]["rate_gbps"] = 100.0

    edit(run, "rg2", bad)
    checks = {finding["check"] for finding in evaluate(run)["nodes"]["rg2"]["findings"]}
    assert checks == {"gpu.nvlink", "nic.rate"}


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


def test_scratch_relative_to_median(run):
    edit(run, "rg5", lambda node: node["scratch"]["data"].update(gbps=0.8))  # above min_GBps 0.2
    assert statuses(run)["rg5"] == "warn"  # severity: warn
    assert statuses(run, Config(scratch=Scratch(read=ScratchRead(severity="fail"))))["rg5"] == "fail"


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
    _, exclude, include = report.write(run, Config(), switches(run), ["fail"])
    assert exclude == "#SBATCH --exclude=rg3\n"
    assert include == "#SBATCH --nodelist=rg[1-2,4-6]\n"
    _, exclude, include = report.write(run, Config(), switches(run), ["fail", "warn"])
    assert exclude == "#SBATCH --exclude=rg[2-4]\n"
    assert include == "#SBATCH --nodelist=rg[1,5-6]\n"
    assert (run / "include.sbatch").read_text() == include


def test_xid_and_dcgm_failures(run):
    edit(run, "rg1", lambda node: node["logs"]["data"]["xid"]["events"].append({"gpu": 1, "xid": 79, "time": "t"}))
    edit(run, "rg6", lambda node: node["dcgm"]["data"]["tests"]["memory"].update(status="Fail", entities={"2": "Fail"}))
    by_host = statuses(run)
    assert by_host["rg1"] == by_host["rg6"] == "fail"


def test_resolve_spread_over_dict():
    assert report.resolve({"net": {"a": {"mtu": 1}, "b": {"mtu": 2}}}, "net[*].mtu") == (1, 2)
    assert report.resolve({}, "gpu.driver_version") is None


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
    assert tests["memory"]["messages"] == ["DBE detected"]
    assert tests["software"]["status"] == "Pass"


def test_dcgm_nonzero_exit_without_failed_test(run):
    edit(run, "rg2", lambda node: node["dcgm"]["data"].update(returncode=226))
    assert statuses(run)["rg2"] == "fail"


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


def test_group_timeout_blames_the_pair_culprit_only(run):
    def hang_rg3_and_group(test):
        if test["kind"] == "group" or (test["kind"] == "pair" and "rg3" in test["nodes"]):
            time_out(test)

    set_tests(run, lambda test: True, hang_rg3_and_group)
    by_host = statuses(run)
    assert by_host["rg3"] == "fail"
    assert by_host["rg1"] == by_host["rg2"] == by_host["rg4"] == "warn"


def test_slow_alltoall_warns_every_node(run):
    set_tests(run, lambda test: True, lambda test: test.update(peak_busbw=2) if test["kind"] == "alltoall" else None)
    assert set(statuses(run).values()) == {"warn"}  # below alltoall_min_busbw_GBps 5
    assert evaluate(run)["medians"]["nccl_alltoall_busbw_GBps"] == 2


def test_alltoall_timeout_without_culprit_excludes_the_group(run):
    set_tests(run, lambda test: True, lambda test: time_out(test) if test["kind"] == "alltoall" else None)
    assert set(statuses(run).values()) == {"fail"}


def test_group_timeout_without_culprit_excludes_the_group(run):
    set_tests(run, lambda test: True, lambda test: time_out(test) if test["kind"] == "group" else None)
    assert set(statuses(run).values()) == {"fail"}
    _, exclude, include = report.write(run, Config(), switches(run), ["fail"])
    assert exclude == "#SBATCH --exclude=rg[1-6]\n"
    assert include == "# no nodes to include\n"
