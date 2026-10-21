import json

import pytest

from healthcheck.util.hostlist import Hostlist
from healthcheck.util.switches import parse_topology
from launcher import topology

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

# sinfo -N -h -o "%N|%t|%f"
SINFO = r"""
rg12501|drng|sapphirerapids,h100mig
rg12502|mix-|sapphirerapids,h100mig
rg21701|mix-|sapphirerapids,h100,nvlink
rg21702|mix-|sapphirerapids,h100,nvlink
rg21703|mix-|sapphirerapids,h100,nvlink
rg31501|mix-|sapphirerapids,h100,nvlink
rg31502|mix-|sapphirerapids,h100,nvlink
rg31503|mix-|sapphirerapids,h100,nvlink
rg31601|mix-|sapphirerapids,h100,nvlink
rg31602|mix-|sapphirerapids,h100,nvlink
rg31603|mix-|sapphirerapids,h100,nvlink
rg31604|drain*|sapphirerapids,h100,nvlink
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
    nodes = topology.parse_sinfo(SINFO)
    usable, skipped = topology.select_nodes(nodes, ["h100", "nvlink"], topology.USABLE_STATES)
    names = [node.name for node in usable]
    assert "rg12501" not in names  # MIG
    assert "rg31601" in names and "rg21701" in names
    assert [node.name for node in skipped] == ["rg31604"]  # "drain*" -> "drain"
    assert all(node.state == "mix" for node in usable)  # "mix-" suffix stripped


def test_filter_and_group_nodes():
    switch_of = {"a": "sw1", "b": "sw1", "c": "sw2"}
    assert topology.filter_nodes("abc", switch_of, None, None) == (("a", "b", "c"), ())
    assert topology.filter_nodes("abc", switch_of, ["a", "c", "z"], None) == (("a", "c"), ("z",))
    assert topology.filter_nodes("abc", switch_of, None, ["sw1"]) == (("a", "b"), ())
    assert topology.group_nodes("abc", switch_of, "switch") == {"sw1": ("a", "b"), "sw2": ("c",)}
    assert topology.group_nodes("abc", switch_of, "cluster") == {"cluster": ("a", "b", "c")}
    assert topology.group_nodes("ab", switch_of, "node") == {"a": ("a",), "b": ("b",)}
