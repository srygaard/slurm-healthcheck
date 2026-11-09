import json
from datetime import datetime

import pytest

from healthcheck.util.cmd import Cmd, ExitCode
from healthcheck.util.hostlist import Hostlist
from healthcheck.util.switches import parse_topology
from launcher import selection, slurm

# slurm.conf topology.conf: nodes only under leaves; spines list switches, and some are commented out
TOPOLOGY_CONF = r"""
SwitchName=ihpc125u25 Nodes=rc[12501-12531],rg[12501-12503]
SwitchName=ihpc218u25 Nodes=rc[21801-21808],rg[21801-21807]
SwitchName=ihpc316u25 Nodes=rg[31501-31503,31601-31609]
#
SwitchName=ihpc207u18 Switches=ihpc125u25,ihpc218u25,ihpc316u25
#SwitchName=ihpc207u16 Switches=ihpc125u25,ihpc218u25,ihpc316u25
"""

# `scontrol show topology`: every level lists the nodes below it
SCONTROL_TOPOLOGY = r"""
SwitchName=ihpc325u25 Level=0 LinkSpeed=1 Nodes=rg[32501-32503]
SwitchName=ihpc207u18 Level=1 LinkSpeed=1 Nodes=rg[32501-32503]
"""

# Rorqual, sinfo -N -h -o "%N|%t|%f|%G"; the MIG gres types are made up, anything but "h100" works
SINFO = r"""
rg12501|drng|sapphirerapids,h100mig|gpu:h100_3g.40gb:2(S:0-1),gpu:h100_2g.20gb:2(S:0-1)
rg12502|mix-|sapphirerapids,h100mig|gpu:h100_3g.40gb:2(S:0-1),gpu:h100_2g.20gb:2(S:0-1)
rg21701|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg21702|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg21703|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31501|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31502|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31503|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31601|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31602|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31603|mix-|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
rg31604|drain*|sapphirerapids,h100,nvlink|gpu:h100:4(S:0-1)
"""

# Trillium GPU: no features; the H200 socket list holds a comma
SINFO_TRILLIUM = r"""
trig0001|mix|(null)|gpu:h100:4(S:2-3)
trig0002|idle|(null)|gpu:h100:4(S:2-3)
trig0063|mix-|(null)|gpu:h200:4(S:1-2,6-7)
trig0101|down*|(null)|gpu:b200:8(S:0-1)
vm-trig0062-1|resv|(null)|gpu:h100:1(S:0)
"""


def test_hostlist():
    assert Hostlist.expand("n[008-010]") == ("n008", "n009", "n010")
    assert str(Hostlist(["n010", "n008", "n009", "n009"])) == "n[008-010]"
    assert Hostlist(["n10", "n9"]) == ("n9", "n10")  # natural order
    assert str(Hostlist(["rg31601"])) == "rg31601"
    assert str(Hostlist()) == ""
    assert json.loads(json.dumps(Hostlist(["b", "a"]))) == ["a", "b"]
    hosts = Hostlist.expand("rc[12501-12503],rg[12501-12503,13601],login1")
    assert Hostlist.expand(str(hosts)) == hosts
    with pytest.raises(ValueError):
        Hostlist.expand("n[10-8]")


def test_parse_topology_skips_comments_and_spines():
    switch_of = parse_topology(TOPOLOGY_CONF)
    assert switch_of["rg31601"] == "ihpc316u25"
    assert switch_of["rg31503"] == "ihpc316u25"
    assert switch_of["rc21801"] == "ihpc218u25"
    assert "ihpc125u25" not in switch_of  # switch names never become nodes
    assert set(switch_of.values()) == {"ihpc125u25", "ihpc218u25", "ihpc316u25"}  # spines have no Nodes=


def test_parse_scontrol_topology():
    assert parse_topology(SCONTROL_TOPOLOGY) == dict.fromkeys(["rg32501", "rg32502", "rg32503"], "ihpc325u25")


def test_select_nodes_full_h100_only():
    nodes = selection.parse_sinfo(SINFO)
    usable, skipped = selection.select_nodes(nodes, "gpu:h100:4", [], selection.USABLE_STATES)
    names = [node.name for node in usable]
    assert "rg12501" not in names  # MIG
    assert "rg31601" in names and "rg21701" in names
    assert [node.name for node in skipped] == ["rg31604"]  # "drain*" -> "drain"
    assert all(node.state == "mix" for node in usable)  # "mix-" suffix stripped
    assert selection.select_nodes(nodes, "gpu:h100:4", ["nvlink"], selection.USABLE_STATES) == (usable, skipped)


def test_select_nodes_by_gres_without_features():
    nodes = selection.parse_sinfo(SINFO_TRILLIUM)
    assert next(node for node in nodes if node.name == "trig0063").gres == {"gpu:h200:4"}
    usable, skipped = selection.select_nodes(nodes, "gpu:h100:4", [], selection.USABLE_STATES)
    assert [node.name for node in usable] == ["trig0001", "trig0002"]  # not the 1-GPU VM, H200 or B200
    assert skipped == []


def test_filter_and_group_nodes():
    switch_of = {"a": "sw1", "b": "sw1", "c": "sw2"}
    assert selection.filter_nodes("abc", switch_of, None, None) == (("a", "b", "c"), ())
    assert selection.filter_nodes("abc", switch_of, ["a", "c", "z"], None) == (("a", "c"), ("z",))
    assert selection.filter_nodes("abc", switch_of, None, ["sw1"]) == (("a", "b"), ())
    assert selection.group_nodes("abc", switch_of, selection.JobPer.SWITCH) == {"sw1": ("a", "b"), "sw2": ("c",)}
    assert selection.group_nodes("abc", switch_of, selection.JobPer.CLUSTER) == {"cluster": ("a", "b", "c")}
    assert selection.group_nodes("ab", switch_of, selection.JobPer.NODE) == {"a": ("a",), "b": ("b",)}


# Rorqual (Slurm 25.11, data_parser v0.0.44), cut down to the fields the parsers read
SQUEUE_JSON = r"""
{"jobs": [
 {"job_id": 25034106, "name": "hc-cluster", "job_state": ["PENDING"], "state_reason": "Priority",
  "start_time": {"set": true, "infinite": false, "number": 1793318464}},
 {"job_id": 25035268, "name": "hc-any1", "job_state": ["PENDING"], "state_reason": "Dependency"}
], "meta": {"plugin": {"data_parser": "data_parser/v0.0.44"}}}
"""

SACCT_JSON = r"""
{"jobs": [
 {"job_id": 24798900, "state": {"current": ["COMPLETED"], "reason": "None"}, "time": {"elapsed": 13},
  "nodes": "rc31801"},
 {"job_id": 25034938, "state": {"current": ["CANCELLED"], "reason": "None"}, "time": {"elapsed": 0},
  "nodes": "None assigned"},
 {"job_id": 25035269, "state": {"current": ["COMPLETED"], "reason": "None"}, "time": {"elapsed": 301},
  "nodes": "rg[31601-31602]"}
]}
"""


def test_squeue(monkeypatch):
    def run(stdout, returncode=0, stderr=""):
        monkeypatch.setattr(slurm, "run", lambda args, **_: Cmd(returncode, stdout, stderr))

    run(SQUEUE_JSON)
    expected_time = datetime.fromtimestamp(1793318464).astimezone()
    time_str = f"{expected_time:%Y-%m-%dT%H:%M:%S}"
    assert slurm.squeue(["--me"]).jobs == [
        slurm.SqueueJob(25034106, "PENDING", expected_time),
        slurm.SqueueJob(25035268, "PENDING"),
    ]
    assert (
        str(slurm.SqueueJob(25034106, "PENDING", expected_time))
        == f"25034106 PENDING (estimated start {time_str})"
    )
    assert str(slurm.SqueueJob(25035268, "PENDING")) == "25035268 PENDING"
    assert str(slurm.SqueueJob(25035268, "RUNNING", expected_time)) == f"25035268 RUNNING (started {time_str})"
    run("", 1, "slurm_load_jobs error: Invalid job id specified\n")  # squeue -j with only finished jobs
    assert slurm.squeue(["-j", "1"]) == ([], 1, "slurm_load_jobs error: Invalid job id specified\n")


def test_sacct_states(monkeypatch):
    def run(stdout):
        monkeypatch.setattr(slurm, "run", lambda args, **_: Cmd(0, stdout, ""))

    run(SACCT_JSON)
    jobs = slurm.sacct_states([24798900, 25034938, 25035269])
    assert jobs[24798900] == {"state": "COMPLETED", "elapsed_s": 13, "nodes": ("rc31801",)}
    assert jobs[25034938]["nodes"] == ()  # never started
    assert jobs[25035269]["nodes"] == ("rg31601", "rg31602")
    run("")
    assert slurm.sacct_states([1]) == {}
    assert slurm.sacct_states([]) == {}  # no sacct call


def test_sbatch_passes_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(slurm, "run", lambda args, **kwargs: calls.append((args, kwargs)) or Cmd(0, "25035268\n", ""))
    assert slurm.sbatch("#!/bin/bash\n", ["--nodes=1"], tmp_path, {"HC_NODES": "rg[31601,31603]"}).job_id == 25035268
    [(args, kwargs)] = calls
    assert args == ["sbatch", "--parsable", "--export=ALL", "--nodes=1"]
    assert kwargs["env"]["HC_NODES"] == "rg[31601,31603]"  # a comma, which --export=NAME=value would split on


def test_sbatch_without_slurm(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    result = slurm.sbatch("#!/bin/bash\n", [], tmp_path, {})
    assert (result.job_id, result.returncode) == (None, ExitCode.NOT_FOUND)


def test_job_template_from_sh(tmp_path):
    script = """#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:15:00
#SBATCH --reservation=maint
#SBATCH --constraint=h100&nvlink  # full nodes
"""
    path = tmp_path / "rorqual.sh"
    path.write_text(script)
    template = slurm.JobTemplate.from_sh(path)
    assert template.text == script
    assert template.gres == slurm.Gres("h100", 4)
    assert str(template.gres) == "h100:4"
    assert template.time == "00:15:00"
    assert template.account == "rrg-bengioy-ad"
    assert template.reservation == "maint"
    assert template.constraint == ["h100", "nvlink"]
    assert template.directives == {
        "account": "rrg-bengioy-ad",
        "gpus-per-node": "h100:4",
        "time": "00:15:00",
        "reservation": "maint",
        "constraint": "h100&nvlink",
    }


def test_job_template_load(monkeypatch, tmp_path):
    (tmp_path / "trillium.sh").write_text("#SBATCH --gpus-per-node=h100:4\n")
    (tmp_path / "rorqual.sh").write_text("#SBATCH --gpus-per-node=b200:8\n")
    monkeypatch.setenv("CC_CLUSTER", "trillium")
    assert slurm.JobTemplate.load(tmp_path).gres == slurm.Gres("h100", 4)
    monkeypatch.setenv("CC_CLUSTER", "rorqual")
    assert slurm.JobTemplate.load(tmp_path).gres == slurm.Gres("b200", 8)
    monkeypatch.setenv("CC_CLUSTER", "missing")
    with pytest.raises(ValueError, match="no job template"):
        slurm.JobTemplate.load(tmp_path)
    monkeypatch.delenv("CC_CLUSTER")
    with pytest.raises(ValueError, match="no job template"):
        slurm.JobTemplate.load(tmp_path)


def test_job_template_gres_count_must_be_int():
    with pytest.raises(ValueError, match="invalid --gpus-per-node count in h100:abc"):
        _ = slurm.JobTemplate("#SBATCH --gpus-per-node=h100:abc\n").gres
    with pytest.raises(ValueError, match="invalid --gpus-per-node count in abc"):
        _ = slurm.JobTemplate("#SBATCH --gpus-per-node=abc\n").gres


def test_job_template_constraint_must_be_and():
    with pytest.raises(ValueError, match="Only AND constraints"):
        _ = slurm.JobTemplate("#SBATCH --constraint=h100|b200\n").constraint


def test_job_template_without_account_or_reservation(tmp_path):
    path = tmp_path / "mila.sh"
    path.write_text("#SBATCH --gpus-per-node=l40s:4\n#SBATCH --time=00:15:00\n")
    template = slurm.JobTemplate.from_sh(path)
    assert template.account is None
    assert template.reservation is None
    assert template.gres == slurm.Gres("l40s", 4)
    assert template.time == "00:15:00"
    assert template.constraint == []
