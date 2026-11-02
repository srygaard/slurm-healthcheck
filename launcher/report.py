"""Step 6: apply the thresholds in healthcheck/config.py to a run's node and group JSONs, then write the report and --exclude header."""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from healthcheck import nccl
from healthcheck.check import Finding
from healthcheck.config import Config, Paths
from healthcheck.node import STEPS
from healthcheck.util.file import read_json
from healthcheck.util.hostlist import ByHost, Hostlist


def evaluate(run: Path, cfg: Config, switch_of: ByHost[str]) -> dict[str, Any]:
    """switch_of maps every node the run targeted to its leaf switch."""
    layout = Paths(root=run)
    jobs = read_json(layout.jobs) or []
    job_of = {node: job for job in jobs for node in job["nodes"]}
    findings: ByHost[list[Finding]] = defaultdict(list)
    results: dict[str, ByHost[Any]] = {name: {} for name in STEPS}
    missing: dict[str, ByHost[Finding]] = {name: {} for name in STEPS}
    step_seconds: ByHost[dict[str, float]] = {}

    groups = [
        group
        for path in sorted(layout.groups.glob("*.json"))
        if (group := read_json(path))
    ]
    hosts = (
        {path.stem for path in layout.nodes.glob("*.json")}
        | set(switch_of)
        | set(job_of)
    )
    for host in sorted(hosts):
        node = read_json(layout.node(host))
        if node is None:
            job = job_of.get(host, {})
            findings[host].append(Finding("fail", "no_report", f"job {job.get('job_id')} {job.get('state', '?')}"))
            continue
        step_seconds[host] = {
            name: node[name]["elapsed_s"]
            for name in STEPS
            if node.get(name)
        }
        for name, check in STEPS.items():
            data, finding = check.unwrap(node, name)
            if finding:
                missing[name][host] = finding
            if data is not None:
                results[name][host] = data

    medians: dict[str, float | None] = {}
    for name, check in STEPS.items():
        # sysinfo reads the GPU, NIC, and homogeneity sections; every other step's section is named after it
        if name == "sysinfo":
            result = check.report(results[name], cfg.gpu, cfg.nic, cfg.homogeneity)
        else:
            result = check.report(results[name], getattr(cfg, name), name)
        medians |= result.medians
        for host, finding in missing[name].items():
            findings[host].append(finding)
        for host, found in result.findings.items():
            findings[host] += found
    nccl_report = nccl.HealthCheck.report(groups, cfg.nccl, "nccl")
    for host, found in nccl_report.findings.items():
        findings[host] += found
    medians |= nccl_report.medians

    nodes = {}
    for host in sorted(set(findings) | set(switch_of)):
        fs = findings.get(host, [])
        status = "fail" if any(finding.severity == "fail" for finding in fs) else "warn" if fs else "pass"
        nodes[host] = {
            "status": status,
            "switch": switch_of.get(host),
            "job_id": job_of.get(host, {}).get("job_id"),
            "findings": [finding._asdict() for finding in fs],
            "elapsed_s": step_seconds.get(host, {}),
        }
    nccl_seconds = {str(group["job_id"]): group.get("elapsed_s") for group in groups}
    return {
        "run": str(run),
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": Counter(node["status"] for node in nodes.values()),
        "medians": medians,
        # elapsed_s from sacct; nccl_elapsed_s is phase B's wall time, from groups/<job_id>.json
        "jobs": [
            {
                "job_id": job["job_id"],
                "label": job.get("label"),
                "nodes": len(job["nodes"]),
                "state": job.get("state"),
                "elapsed_s": job.get("elapsed_s"),
                "nccl_elapsed_s": nccl_seconds.get(str(job["job_id"])),
            }
            for job in jobs
        ],
        "nodes": nodes,
    }


def format_text(report: dict[str, Any]) -> str:
    lines = [
        f"Run {report['run']}  ({report['generated']})",
        "Summary: " + ", ".join(f"{status} {count}" for status, count in sorted(report["summary"].items())),
        "",
        "Medians (for calibrating healthcheck/config.py):",
    ]
    lines += [f"  {name:<24} {median}" for name, median in report["medians"].items()]
    lines += ["", f"{'job':<10} {'label':<14} {'nodes':>5} {'state':<10} {'elapsed_s':>9} {'nccl_s':>7}"]
    lines += [
        f"{job['job_id']:<10} {job['label'] or '-':<14} {job['nodes']:>5} {job['state'] or '-':<10} "
        f"{job['elapsed_s'] if job['elapsed_s'] is not None else '-':>9} "
        f"{job['nccl_elapsed_s'] if job['nccl_elapsed_s'] is not None else '-':>7}"
        for job in report["jobs"]
    ]
    lines += ["", f"{'node':<10} {'switch':<12} {'job':<10} {'status':<6} findings"]
    for host, node in report["nodes"].items():
        fs = node["findings"] or [{"severity": "", "check": "", "detail": ""}]
        for index, finding in enumerate(fs):
            prefix = (
                f"{host:<10} {node['switch'] or '-':<12} {node['job_id'] or '-':<10} {node['status']:<6}"
                if index == 0
                else " " * 41
            )
            lines.append(f"{prefix} {finding['severity']:<4} {finding['check']:<16} {finding['detail']}".rstrip())
    return "\n".join(lines) + "\n"


def exclude_header(report: dict[str, Any], exclude_on: list[str]) -> str:
    bad = [host for host, node in report["nodes"].items() if node["status"] in exclude_on]
    return f"#SBATCH --exclude={Hostlist(bad)}\n" if bad else "# no nodes to exclude\n"


def include_header(report: dict[str, Any], exclude_on: list[str]) -> str:
    """The run's other nodes, for a job that should run on exactly those."""
    good = [host for host, node in report["nodes"].items() if node["status"] not in exclude_on]
    return f"#SBATCH --nodelist={Hostlist(good)}\n" if good else "# no nodes to include\n"


def write(
    run: Path,
    cfg: Config,
    switch_of: ByHost[str],
    exclude_on: list[str],
) -> tuple[dict[str, Any], str, str]:
    """Returns the report and the exclude and include headers."""
    report = evaluate(run, cfg, switch_of)
    exclude = exclude_header(report, exclude_on)
    include = include_header(report, exclude_on)
    (run / "report.json").write_text(json.dumps(report, indent=1))
    (run / "report.txt").write_text(format_text(report))
    (run / "exclude.sbatch").write_text(exclude)
    (run / "include.sbatch").write_text(include)
    return report, exclude, include
