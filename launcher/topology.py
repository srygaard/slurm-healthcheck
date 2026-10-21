"""Find the healthcheck candidate nodes and group them into jobs."""

import subprocess
from collections.abc import Iterable, Mapping
from typing import Literal, NamedTuple

from healthcheck.util.hostlist import Hostlist

type Mode = Literal["cluster", "switch", "node"]

STATE_SUFFIXES = "*~#!%$@^-+"

# `sinfo %t` states, lowercased with the STATE_SUFFIXES flags stripped, in which a node can run a healthcheck job.
# Busy nodes only make it wait; any other state could leave a --nodelist job pending for good, so it is skipped.
USABLE_STATES = {"idle", "mix", "alloc", "comp", "plnd"}


class Node(NamedTuple):
    name: str
    state: str  # `sinfo %t`, lowercased with the STATE_SUFFIXES flags stripped
    features: frozenset[str]


def parse_sinfo(text: str) -> list[Node]:
    """Parse `sinfo -N -h -o "%N|%t|%f"`; nodes in several partitions appear once."""
    nodes: dict[str, Node] = {}
    for line in text.splitlines():
        if line.count("|") != 2:
            continue
        name, state, features = (field.strip() for field in line.split("|"))
        nodes[name] = Node(name, state.rstrip(STATE_SUFFIXES).lower(), frozenset(features.split(",")))
    return sorted(nodes.values())


def sinfo_nodes() -> list[Node]:
    out = subprocess.run(["sinfo", "-N", "-h", "-o", "%N|%t|%f"], capture_output=True, text=True, check=True).stdout
    return parse_sinfo(out)


def select_nodes(
    nodes: Iterable[Node],
    features: Iterable[str],
    usable_states: Iterable[str],
) -> tuple[list[Node], list[Node]]:
    """Split nodes having all features into (usable, skipped by state)."""
    required = set(features)
    usable = set(usable_states)
    matching = [node for node in nodes if required <= node.features]
    return [node for node in matching if node.state in usable], [node for node in matching if node.state not in usable]


def filter_nodes(
    nodes: Iterable[str],
    switch_of: Mapping[str, str],
    names: Iterable[str] | None,
    switches: Iterable[str] | None,
) -> tuple[Hostlist, Hostlist]:
    """The nodes among names and under switches (None keeps all), and the names that aren't among nodes."""
    nodes = Hostlist(nodes)
    wanted = set(nodes) if names is None else set(names)
    under = None if switches is None else set(switches)
    kept = Hostlist(node for node in nodes if node in wanted and (under is None or switch_of.get(node) in under))
    return kept, Hostlist(wanted - set(nodes))


def group_nodes(nodes: Iterable[str], switch_of: Mapping[str, str], mode: Mode) -> dict[str, Hostlist]:
    """Nodes per job label: one job for the cluster, one per leaf switch, or one per node."""
    groups: dict[str, list[str]] = {}
    for node in nodes:
        label = {"cluster": "cluster", "switch": switch_of.get(node, "unknown"), "node": node}[mode]
        groups.setdefault(label, []).append(node)
    return {label: Hostlist(members) for label, members in groups.items()}
