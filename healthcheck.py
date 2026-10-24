#!/usr/bin/env -S uv run
"""Submit cluster health checks to SLURM, then report which nodes to --exclude."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from healthcheck.config import Paths, cfg
from healthcheck.util.file import read_json
from healthcheck.util.hostlist import Hostlist
from healthcheck.util.switches import node_switches
from launcher import report, selection
from launcher.slurm import detect_cluster, my_queued, sacct_states, sbatch, squeue, wait_for, with_dependency

REPO = Path(__file__).parent



class Ansi(StrEnum):
    PURPLE = "\033[35m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"


def base_dir() -> Path:
    """.opt/, .uv-cache/, .venv/, diskprobe/, env.json, results/; jobs get it as $HC_BASE."""
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit(f"{Ansi.RED}$SCRATCH is not set{Ansi.RESET}")
    base = Path(scratch) / "slurm-healthcheck"
    base.mkdir(exist_ok=True)  # sbatch runs from it, so it must exist before the first setup
    return base


def venv_dir(base: Path) -> Path:
    """Shared venv keyed by the lock, as in the slurm-launcher README recipe."""
    digest = hashlib.sha256((REPO / "uv.lock").read_bytes() + (REPO / ".python-version").read_bytes())
    return base / ".venv" / digest.hexdigest()[:16]


def job_env(base: Path, run_id: int | None, extra: Mapping[str, object] | None = None) -> dict[str, str]:
    """Variables every job gets, plus extra; the templates and healthcheck read them."""
    env: dict[str, object] = {"HC_BASE": base, "HC_VENV": venv_dir(base), "HC_CODE": REPO, "PYTHONPATH": REPO}
    if run_id:
        env["HC_RUN_ID"] = run_id
    env.update(extra or {})
    return {name: str(value) for name, value in env.items()}


def jobs_dir(base: Path, run_id: int | str) -> Path:
    """results/<run>/jobs/: each job's saved script and its stdout/stderr."""
    return base / "results" / str(run_id) / "jobs"


def sbatch_options(base: Path, label: str, run_id: int | None) -> list[str]:
    """Per-job options; they override the template's #SBATCH directives."""
    logs = jobs_dir(base, run_id or "%j")  # sbatch fills in %j
    return [f"--job-name=hc-{label}", f"--output={logs}/%j-stdout.txt", f"--error={logs}/%j-stderr.txt"]


def template() -> str:
    """slurm/templates/<cluster>.sh: the resources of every job on this cluster, the probe's and the healthcheck's."""
    cluster = detect_cluster()
    templates = REPO / "slurm" / "templates"
    path = templates / f"{cluster}.sh"
    if not cluster or not path.is_file():
        known = ", ".join(sorted(found.stem for found in templates.glob("*.sh")))
        raise SystemExit(
            f"{Ansi.RED}No job template for cluster {cluster or '?'}: set $CC_CLUSTER to one of {known}{Ansi.RESET}"
        )
    return path.read_text()


def job_script(command: str) -> str:
    """The cluster's template followed by command."""
    return f"{template().rstrip()}\n\n{command}\n"


def sbatch_directives() -> dict[str, str]:
    """The template's `#SBATCH --option=value` lines as {option: value}."""
    return dict(re.findall(r"^#SBATCH\s+--([\w-]+)=(\S+)", template(), re.MULTILINE))


def submit(
    command: str,
    base: Path,
    label: str,
    options: Sequence[str] = (),
    run_id: int | None = None,
) -> int | None:
    """sbatch the cluster's template followed by command, printing why a submission failed.

    The job logs to results/<run_id or its own job id>/jobs/.
    """
    options = [*sbatch_options(base, label, run_id), *options]
    result = sbatch(job_script(command), options, cwd=base, env=job_env(base, run_id))
    if not result.job_id:
        print(f"{Ansi.RED}{result.stderr or result.stdout}{Ansi.RESET}")
        return None
    # Slurm doesn't create the log directory; the job is still pending, so this is in time
    jobs_dir(base, run_id or result.job_id).mkdir(parents=True, exist_ok=True)
    return result.job_id


def cmd_setup() -> int:
    """Build on the login node, which can reach git and PyPI, then submit the probe job."""
    base = base_dir()
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(f"{Ansi.RED}uv not found on PATH{Ansi.RESET}")
    env = job_env(base, None, {"HC_UV": uv})
    if subprocess.run(["bash", REPO / "slurm" / "scripts" / "build.sh"], env=os.environ | env).returncode:
        print(f"{Ansi.RED}Build failed{Ansi.RESET}")
        return 1
    job_id = submit('exec "$HC_VENV/bin/python" -m healthcheck.probe', base, "probe", options=["--nodes=1"])
    if not job_id:
        return 1
    print(f"probe job {Ansi.PURPLE}{job_id}{Ansi.RESET}: log {base}/results/{job_id}/jobs/{job_id}-stdout.txt")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    base = base_dir()
    venv = venv_dir(base)
    env_json = Paths(base=base).env
    # The jobs read env.json when they start, so they wait for a probe still in the queue to rewrite it
    probes = my_queued("hc-probe")
    if not (venv.is_dir() and (env_json.is_file() or probes)):
        raise SystemExit(
            f"{Ansi.RED}Missing {venv if not venv.is_dir() else env_json}: "
            f"run `uv run healthcheck.py setup` first{Ansi.RESET}"
        )
    if probes:
        print(f"Waiting for probe job(s) {', '.join(map(str, probes))} to succeed")
    dependency = with_dependency(args.dependency, probes)

    if args.any:
        # Slurm picks the nodes, so the job starts on the first free ones; the report learns them from sacct
        switch_of: dict[str, str] = {}
        groups = {f"any{args.any}": Hostlist()}
        print(f"1 job on any {args.any} free node(s)")
    else:
        # The nodes the template's jobs can run on
        directives = sbatch_directives()
        constraint = directives.get("constraint", "")
        if re.search(r"[|\[\]*()]", constraint):
            raise SystemExit(f"{Ansi.RED}Only AND constraints (a&b) can select nodes, not {constraint}{Ansi.RESET}")
        selected, skipped = selection.select_nodes(
            selection.sinfo_nodes(directives.get("partition")),
            f"gpu:{directives['gpus-per-node']}",
            constraint.split("&") if constraint else [],
            # A MAINT reservation's nodes show as maint; the template's --reservation can still run on them
            selection.USABLE_STATES | ({"maint"} if "reservation" in directives else set()),
        )
        if skipped:
            states = Counter(node.state for node in skipped)
            counts = ", ".join(f"{state} {count}" for state, count in states.items())
            print(f"Skipping {len(skipped)} nodes by state ({counts}): {Hostlist(node.name for node in skipped)}")
        all_switches = node_switches()
        if not all_switches and args.switches:
            raise SystemExit(f"{Ansi.RED}Slurm has no switch topology here, so --switches matches no node{Ansi.RESET}")
        if not all_switches and args.job_per == selection.JobPer.SWITCH:
            # topology/flat, e.g. Trillium's non-blocking fabric: every node is under the same switch
            print("No switch topology: one job across all nodes, as --job-per cluster")
            args.job_per = selection.JobPer.CLUSTER
        nodes, missing = selection.filter_nodes(
            (node.name for node in selected),
            all_switches,
            args.nodes,
            args.switches.split(",") if args.switches else None,
        )
        if missing:
            print(f"{Ansi.YELLOW}Not eligible or unknown: {missing}{Ansi.RESET}")
        if not nodes:
            raise SystemExit(f"{Ansi.RED}No eligible nodes{Ansi.RESET}")
        switch_of = {node: all_switches.get(node, "unknown") for node in nodes}
        groups = selection.group_nodes(nodes, switch_of, args.job_per)
        print(f"{len(nodes)} nodes in {len(groups)} {args.job_per} job(s):")
        for label, members in groups.items():
            print(f"  {label:<14} {len(members):>3}  {members}")

    jobs: list[dict[str, Any]] = []
    for label, members in groups.items():
        run_id = jobs[0]["job_id"] if jobs else None  # the first job's id names the run dir; later jobs get HC_RUN_ID
        options = [f"--nodes={len(members) or args.any}"]
        options += [f"--nodelist={members}"] if members else []
        options += [f"--dependency={dependency}"] if dependency else []
        if job_id := submit('exec "$HC_VENV/bin/python" -m healthcheck.batch', base, label, options, run_id):
            jobs.append({"job_id": job_id, "label": label, "nodes": members})
    if not jobs:
        print(f"Submitted 0/{len(groups)} jobs")
        return 1
    run = Paths(base=base, root=(base / "results" / str(jobs[0]["job_id"])).resolve())
    run.jobs.write_text(json.dumps(jobs, indent=1))
    # The report lists every node here, so one that never writes nodes/<host>.json shows as no_report
    by_switch: dict[str, list[str]] = {}
    for node, switch in switch_of.items():
        by_switch.setdefault(switch, []).append(node)
    run.switches.write_text(json.dumps(by_switch, indent=1))
    print(f"Submitted {len(jobs)}/{len(groups)} jobs; run dir {Ansi.PURPLE}{run.root}{Ansi.RESET}")

    if args.no_wait:
        print(f"Report when the jobs finish: {Ansi.GREEN}uv run healthcheck.py report {run.root}{Ansi.RESET}")
        return 0
    try:
        wait_for([job["job_id"] for job in jobs])
    except KeyboardInterrupt:
        print(
            "\nStopped waiting; the jobs still run. Report when they finish: "
            f"{Ansi.GREEN}uv run healthcheck.py report {run.root}{Ansi.RESET}"
        )
        return 130
    return cmd_report(run.root, args.exclude_on.split(","))


def cmd_report(run_dir: Path, exclude_on: list[str]) -> int:
    run = Paths(root=run_dir.resolve())

    jobs = read_json(run.jobs) or []
    # A job still in the queue hasn't written all its results: its nodes would show as failed in exclude.sbatch
    job_ids = ",".join(str(job["job_id"]) for job in jobs)
    if job_ids and (queued := squeue(["-j", job_ids]).jobs):
        states = ", ".join(f"{job_id} {state}" for job_id, state in sorted(queued.items()))
        print(f"{Ansi.YELLOW}Not finished: job {states}{Ansi.RESET}")
        print(f"Report when the jobs finish: {Ansi.GREEN}uv run healthcheck.py report {run.root}{Ansi.RESET}")
        return 1
    states = sacct_states([job["job_id"] for job in jobs])
    for job in jobs:
        if sacct := states.get(job["job_id"]):
            job["state"] = sacct["state"]
            job["elapsed_s"] = sacct["elapsed_s"]
            if not job["nodes"]:  # --any: the nodes are known only once the job started
                job["nodes"] = sacct["nodes"]
    run.jobs.write_text(json.dumps(jobs, indent=1))
    # switches.json was written at submit, so look up the switches of nodes learned from sacct here
    by_switch = read_json(run.switches) or {}
    switch_of = {node: switch for switch, nodes in by_switch.items() for node in nodes}
    if missing := [node for job in jobs for node in job["nodes"] if node not in switch_of]:
        topo = node_switches()
        switch_of |= {node: topo.get(node, "unknown") for node in missing}

    result, exclude, include = report.write(run.root, cfg, switch_of, exclude_on)
    print(report.format_text(result))
    color = Ansi.RED if "--exclude" in exclude else Ansi.GREEN
    print(f"{color}{exclude}{Ansi.RESET}{include}")
    print(f"Written: {run.root}/report.json, report.txt, exclude.sbatch, include.sbatch")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = argparse.ArgumentDefaultsHelpFormatter
    exclude_on_help = 'Report statuses to --exclude (comma separated); "fail,warn" also excludes suspects'

    sub.add_parser(
        "setup",
        help="Build on the login node (nccl-tests, read-test files, venv), then a GPU probe job "
        "(how to run DCGM, which srun MPI plugin works)",
    )

    run_parser = sub.add_parser("run", formatter_class=defaults, help="Submit health checks, wait, and report")
    run_parser.add_argument(
        "-j",
        "--job-per",
        type=selection.JobPer,
        choices=list(selection.JobPer),
        default=selection.JobPer.SWITCH,
        help="One job for all nodes, one per leaf switch, or one per node",
    )
    run_parser.add_argument(
        "-n", "--nodes", type=Hostlist.expand, help="Only these nodes (hostlist, e.g. rg[31601-31609])"
    )
    run_parser.add_argument("-s", "--switches", help="Only nodes under these leaf switches (comma separated)")
    run_parser.add_argument(
        "-a",
        "--any",
        type=int,
        metavar="N",
        help="One job on any N free nodes chosen by Slurm, instead of enumerated nodes",
    )
    run_parser.add_argument(
        "-d",
        "--dependency",
        default=None,
        help="sbatch --dependency, e.g. afterok:12345; a queued probe job is added with afterok",
    )
    run_parser.add_argument("--no-wait", action="store_true", help="Submit and exit; run `report` later")
    run_parser.add_argument("-e", "--exclude-on", default="fail", help=exclude_on_help)

    report_parser = sub.add_parser(
        "report", formatter_class=defaults, help="Re-evaluate a run, e.g. after editing healthcheck/config.py"
    )
    report_parser.add_argument("run_dir", type=Path, help="A results/<job_id> directory")
    report_parser.add_argument("-e", "--exclude-on", default="fail", help=exclude_on_help)

    args = parser.parse_args()
    if getattr(args, "any", None) and (args.nodes or args.switches):
        parser.error("--any picks nodes itself; drop --nodes/--switches")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "setup":
        return cmd_setup()
    if args.command == "run":
        return cmd_run(args)
    return cmd_report(args.run_dir, args.exclude_on.split(","))


if __name__ == "__main__":
    raise SystemExit(main())
