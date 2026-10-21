#!/usr/bin/env -S uv run
"""Submit cluster health checks to SLURM, then report which nodes to --exclude."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from healthcheck.config import Paths, cfg
from healthcheck.util.file import read_json
from healthcheck.util.hostlist import Hostlist
from healthcheck.util.switches import node_switches
from launcher import report, topology
from launcher.sbatch import sbatch
from launcher.slurm import detect_cluster, sacct_states, wait_for

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


def job_env(base: Path, run_id: int | None, extra: Mapping[str, object] | None) -> dict[str, object]:
    """Variables every job gets, plus extra; the templates and healthcheck read them."""
    env: dict[str, object] = {"HC_BASE": base, "HC_VENV": venv_dir(base), "HC_CODE": REPO, "PYTHONPATH": REPO}
    if run_id:
        env["HC_RUN_ID"] = run_id
    env.update(extra or {})
    if with_comma := [name for name, value in env.items() if "," in str(value)]:
        raise SystemExit(f"{Ansi.RED}sbatch --export can't pass a value with a comma: {with_comma}{Ansi.RESET}")
    return env


def jobs_dir(base: Path, run_id: int | str) -> Path:
    """results/<run>/jobs/: each job's saved script and its stdout/stderr."""
    return base / "results" / str(run_id) / "jobs"


def sbatch_options(base: Path, label: str, run_id: int | None, env: Mapping[str, object]) -> list[str]:
    """Per-job options; they override the template's #SBATCH directives. --export sets the environment before
    the template's first line, which sources slurm/scripts/.env."""
    logs = jobs_dir(base, run_id or "%j")  # sbatch fills in %j
    exports = ",".join(["ALL", *(f"{name}={value}" for name, value in env.items())])
    return [
        f"--job-name=hc-{label}",
        f"--output={logs}/%j-stdout.txt",
        f"--error={logs}/%j-stderr.txt",
        f"--export={exports}",
    ]


def job_script(name: str, command: str) -> str:
    """slurm/templates/<name>.<cluster>.sh followed by command."""
    template = REPO / "slurm" / "templates" / f"{name}.{detect_cluster()}.sh"
    return f"{template.read_text().rstrip()}\n\n{command}\n"


def submit(
    name: str,
    command: str,
    base: Path,
    label: str,
    exports: Mapping[str, object] | None = None,
    options: Sequence[str] = (),
    run_id: int | None = None,
) -> int | None:
    """sbatch the <name> template followed by command, printing why a submission failed.

    The job logs to results/<run_id or its own job id>/jobs/.
    """
    env = job_env(base, run_id, exports)
    options = [*sbatch_options(base, label, run_id, env), *options]
    script = job_script(name, command)
    result = sbatch(script, options, cwd=base)
    if not result.job_id:
        print(f"{Ansi.RED}{result.stderr or result.stdout}{Ansi.RESET}")
        return None
    # Slurm doesn't create the log directory; the job is still pending, so this is in time
    jobs_dir(base, run_id or result.job_id).mkdir(parents=True, exist_ok=True)
    return result.job_id


def cmd_setup() -> int:
    """Submit the build job, then the probe job once the build succeeds."""
    base = base_dir()
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(f"{Ansi.RED}uv not found on PATH{Ansi.RESET}")
    commands = {
        "build": 'bash "$HC_CODE/slurm/scripts/build.sh"',
        "probe": 'exec "$HC_VENV/bin/python" -m healthcheck.probe',
    }
    job_id = None
    for name, command in commands.items():
        job_id = submit(name, command, base, name, {"HC_UV": uv}, [f"--dependency=afterok:{job_id}"] if job_id else [])
        if not job_id:
            return 1
        print(f"{name} job {Ansi.PURPLE}{job_id}{Ansi.RESET}: log {base}/results/{job_id}/jobs/{job_id}-stdout.txt")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    base = base_dir()
    venv = venv_dir(base)
    env_json = Paths(base=base).env
    if not (venv.is_dir() and env_json.is_file()):
        raise SystemExit(
            f"{Ansi.RED}Missing {venv if not venv.is_dir() else env_json}: "
            f"run `uv run healthcheck.py setup` first{Ansi.RESET}"
        )

    if args.any:
        # Slurm picks the nodes, so the job starts on the first free ones; the report learns them from sacct
        switch_of: dict[str, str] = {}
        groups = {f"any{args.any}": Hostlist()}
        print(f"1 job on any {args.any} free node(s)")
    else:
        selected, skipped = topology.select_nodes(
            topology.sinfo_nodes(), args.features.split(","), topology.USABLE_STATES
        )
        if skipped:
            states = Counter(node.state for node in skipped)
            counts = ", ".join(f"{state} {count}" for state, count in states.items())
            print(f"Skipping {len(skipped)} nodes by state ({counts}): {Hostlist(node.name for node in skipped)}")
        all_switches = node_switches()
        nodes, missing = topology.filter_nodes(
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
        groups = topology.group_nodes(nodes, switch_of, args.mode)
        print(f"{len(nodes)} nodes in {len(groups)} {args.mode} job(s):")
        for label, members in groups.items():
            print(f"  {label:<14} {len(members):>3}  {members}")

    jobs: list[dict[str, Any]] = []
    for label, members in groups.items():
        run_id = jobs[0]["job_id"] if jobs else None  # the first job's id names the run dir; later jobs get HC_RUN_ID
        options = [f"--nodes={len(members) or args.any}"]
        options += [f"--nodelist={members}"] if members else []
        options += [f"--dependency={args.dependency}"] if args.dependency else []
        if job_id := submit(
            "healthcheck",
            'exec "$HC_VENV/bin/python" -m healthcheck.batch',
            base,
            label,
            options=options,
            run_id=run_id,
        ):
            jobs.append({"job_id": job_id, "label": label, "nodes": members})
    if not jobs:
        print(f"Submitted 0/{len(groups)} jobs")
        return 1
    run = Paths(base=base, root=(base / "results" / str(jobs[0]["job_id"])).resolve())
    run.jobs.write_text(json.dumps(jobs, indent=1))
    # The report lists every node here, so one that never writes nodes/<host>.json shows as no_report
    run.switches.write_text(json.dumps(switch_of, indent=1))
    print(f"Submitted {len(jobs)}/{len(groups)} jobs; run dir {Ansi.PURPLE}{run.root}{Ansi.RESET}")

    if args.no_wait:
        print(f"Later: uv run healthcheck.py report {run.root}")
        return 0
    try:
        wait_for([job["job_id"] for job in jobs])
    except KeyboardInterrupt:
        print(f"\nJobs keep running. Later: uv run healthcheck.py report {run.root}")
        return 130
    return cmd_report(run.root, args.exclude_on.split(","))


def cmd_report(run_dir: Path, exclude_on: list[str]) -> int:
    run = Paths(root=run_dir.resolve())

    jobs = read_json(run.jobs) or []
    states = sacct_states([job["job_id"] for job in jobs])
    for job in jobs:
        if sacct := states.get(job["job_id"]):
            job["state"] = sacct["state"]
            job["elapsed_s"] = sacct["elapsed_s"]
            if not job["nodes"]:  # --any: the nodes are known only once the job started
                job["nodes"] = sacct["nodes"]
    run.jobs.write_text(json.dumps(jobs, indent=1))
    # switches.json was written at submit, so look up the switches of nodes learned from sacct here
    switch_of = read_json(run.switches) or {}
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
        help="CPU build job (nccl-tests, read-test files, venv), then a GPU probe job "
        "(how to run DCGM, which srun MPI plugin works)",
    )

    run_parser = sub.add_parser("run", formatter_class=defaults, help="Submit health checks, wait, and report")
    run_parser.add_argument(
        "-m",
        "--mode",
        choices=["cluster", "switch", "node"],
        default="switch",
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
        "-f",
        "--features",
        default="h100,nvlink",
        help="Only nodes with all these features (comma separated); MIG nodes have h100mig instead of h100",
    )
    run_parser.add_argument("-d", "--dependency", default=None, help="sbatch --dependency, e.g. afterok:12345")
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
