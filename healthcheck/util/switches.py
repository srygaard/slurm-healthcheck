"""Each node's leaf switch, from Slurm's network topology."""

from healthcheck.util.cmd import run
from healthcheck.util.hostlist import Hostlist


def parse_topology(text: str) -> dict[str, str]:
    """Map each node to its leaf switch from "SwitchName=X ... Nodes=<hostlist>" lines.

    topology.conf lists nodes only under leaves; `scontrol show topology` lists them under every level too.
    """
    switch_of: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        fields = dict(field.split("=", 1) for field in line.split() if "=" in field)
        if "SwitchName" in fields and "Nodes" in fields and fields.get("Level", "0") == "0":
            for node in Hostlist.expand(fields["Nodes"]):
                switch_of[node] = fields["SwitchName"]
    return switch_of


def node_switches() -> dict[str, str]:
    """From `scontrol show topology`, which works on compute nodes too; {} if the cluster has no topology."""
    out = run(["scontrol", "show", "topology"], timeout=30)
    return parse_topology(out.stdout) if out.returncode == 0 else {}
