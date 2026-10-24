"""Step 6: apply the thresholds in healthcheck/config.py to a run's node and group JSONs, then write the report and --exclude header."""

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

from healthcheck.config import Config, Dcgm, Gpu, LocalTmp, Logs, Nccl, Nic, Paths, ScratchRead
from healthcheck.util.file import read_json
from healthcheck.util.hostlist import Hostlist
from healthcheck.util.status import Status, ok

type Severity = Literal["fail", "warn"]

STEPS = ("sysinfo", "scratch", "local_tmp", "dcgm", "logs")  # the node JSON's steps, each with elapsed_s


class Finding(NamedTuple):
    severity: Severity
    check: str
    detail: str


def resolve(data: Any, path: str) -> Any:
    """
    Values at a dotted key path; "[*]" spans a list or dict, giving a tuple.

    >>> resolve({"nic": {"ib": [{"fw": "1"}, {"fw": "2"}]}}, "nic.ib[*].fw")
    ('1', '2')
    """
    head, _, rest = path.partition(".")
    spread = head.endswith("[*]")
    key = head.removesuffix("[*]")
    if not isinstance(data, dict) or key not in data:
        return None
    value = data[key]
    if spread:
        items = value.values() if isinstance(value, dict) else value or []
        return tuple(resolve(item, rest) if rest else item for item in items)
    return resolve(value, rest) if rest else value


def _median(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return round(statistics.median(values), 4) if values else None


def _step(node: dict[str, Any], name: str, findings: list[Finding]) -> dict[str, Any] | None:
    """The step's data, or None after recording why it is missing."""
    result = node.get(name)
    if result is None:
        findings.append(Finding("fail", name, "not run"))
        return None
    if not ok(result):
        findings.append(Finding("fail", name, f"{result['status']}: {result.get('error', '')}".strip(": ")))
        return None
    return result["data"]


def check_sysinfo(data: dict[str, Any], gpu_cfg: Gpu, nic_cfg: Nic) -> list[Finding]:
    out = []
    gpu = data["gpu"]
    if gpu["count"] != gpu_cfg.count:
        out.append(Finding("fail", "gpu.count", f"{gpu['count']} GPUs, expected {gpu_cfg.count}"))
    for device in gpu["devices"]:
        index = device["index"]
        if gpu_cfg.require_ecc and device["ecc_enabled"] is False:
            out.append(Finding("fail", "gpu.ecc", f"GPU{index} ECC disabled"))
        if device["mig_enabled"]:
            out.append(Finding("fail", "gpu.mig", f"GPU{index} MIG enabled"))
        if (device["pcie_gen_max"] or 0) < gpu_cfg.min_pcie_gen:
            out.append(
                Finding(
                    "fail", "gpu.pcie", f"GPU{index} max PCIe gen {device['pcie_gen_max']} < {gpu_cfg.min_pcie_gen}"
                )
            )
        if (device["pcie_width"] or 0) < gpu_cfg.min_pcie_width:
            out.append(
                Finding("fail", "gpu.pcie", f"GPU{index} PCIe x{device['pcie_width']} < x{gpu_cfg.min_pcie_width}")
            )
        if device["nvlinks_active"] < gpu_cfg.min_nvlinks:
            out.append(
                Finding("fail", "gpu.nvlink", f"GPU{index} {device['nvlinks_active']} NVLinks < {gpu_cfg.min_nvlinks}")
            )
    for counters in data.get("counters", []):
        index = counters["index"]
        if (counters["ecc_volatile"]["uncorrected"] or 0) > gpu_cfg.max_volatile_uncorrected_ecc:
            out.append(
                Finding("fail", "gpu.ecc", f"GPU{index} {counters['ecc_volatile']['uncorrected']} uncorrected ECC")
            )
        if rows := counters["remapped_rows"]:
            if rows["failure"] > gpu_cfg.max_remap_failures:
                out.append(Finding("fail", "gpu.remap", f"GPU{index} row remap failure"))
            if rows["pending"] > gpu_cfg.max_remap_pending:
                out.append(Finding("fail", "gpu.remap", f"GPU{index} row remap pending (needs reset)"))

    active = [
        (dev["name"], port)
        for dev in data["nic"]["ib"]
        for port in dev["ports"]
        if port["state"] == "ACTIVE" and port["link_layer"] == nic_cfg.link_layer
    ]
    if len(active) < nic_cfg.min_active_ib_ports:
        out.append(
            Finding(
                "fail", "nic.ports", f"{len(active)} active {nic_cfg.link_layer} ports < {nic_cfg.min_active_ib_ports}"
            )
        )
    for name, port in active:
        if (port["rate_gbps"] or 0) < nic_cfg.min_rate_Gbps:
            out.append(
                Finding("fail", "nic.rate", f"{name}/{port['port']} {port['rate']} < {nic_cfg.min_rate_Gbps} Gb/s")
            )
    return out


def check_scratch_read(data: dict[str, Any], cfg: ScratchRead) -> list[Finding]:
    if not ok(data):
        return [Finding("warn", "scratch.read", f"{data['status']}: {data['path']} (run setup)")]
    if data["gbps"] < cfg.min_GBps:
        return [Finding(cfg.severity, "scratch.read", f"read {data['gbps']} < {cfg.min_GBps} GB/s")]
    return []


def check_local_tmp(data: dict[str, Any], cfg: LocalTmp) -> list[Finding]:
    out = []
    for path, tmp in data["paths"].items():
        if tmp["free_gib"] < cfg.min_free_gib:
            out.append(Finding("fail", "local_tmp.free", f"{path} {tmp['free_gib']} GiB free < {cfg.min_free_gib}"))
        if "error" in tmp:
            out.append(Finding("fail", "local_tmp.rw", f"{path}: {tmp['error']}"))
            continue
        if "same_device_as" in tmp:  # benchmarked under that path
            continue
        if tmp["write"]["gbps"] < cfg.rw.min_write_GBps:
            out.append(
                Finding("fail", "local_tmp.rw", f"{path} write {tmp['write']['gbps']} GB/s < {cfg.rw.min_write_GBps}")
            )
        if tmp["read"]["gbps"] < cfg.rw.min_read_GBps:
            out.append(
                Finding("fail", "local_tmp.rw", f"{path} read {tmp['read']['gbps']} GB/s < {cfg.rw.min_read_GBps}")
            )
    for name, disk in data["smart"].get("disks", {}).items():
        if cfg.fail_on_smart and disk.get("passed") is False:
            out.append(Finding("fail", "local_tmp.smart", f"/dev/{name} SMART health failed"))
    return out


def check_dcgm(data: dict[str, Any], cfg: Dcgm) -> list[Finding]:
    if data["status"] == Status.SKIPPED:
        return [Finding("warn", "dcgm", f"not run (dcgm={data['mode']})")] if cfg.enabled else []
    out = []
    for name, test in data["tests"].items():
        severity: Severity | None = (
            "fail" if test["status"] in cfg.fail_on else "warn" if test["status"] in cfg.warn_on else None
        )
        if severity:
            bad = [entity for entity, status in test["entities"].items() if status == test["status"]]
            detail = f"{name} {test['status']} on {','.join(bad)}"
            out.append(Finding(severity, "dcgm", f"{detail}: {test['messages'][0]}" if test["messages"] else detail))
    if not out and data.get("returncode"):
        out.append(Finding("fail", "dcgm", f"dcgmi diag exited {data['returncode']} with no failed test"))
    return out


def check_logs(data: dict[str, Any], cfg: Logs) -> list[Finding]:
    out = []
    for event in data["xid"].get("events", []):
        out.append(Finding("fail", "logs.xid", f"GPU{event['gpu']} Xid {event['xid']} at {event['time']}"))
    for delta in data.get("counter_deltas") or []:
        for key, increase in delta["increased"].items():
            if key.startswith("ecc_aggregate"):
                continue  # rises with ecc_volatile, which is reported
            nvlink_over = key.startswith("nvlink_errors") and increase > cfg.max_nvlink_error_delta
            if key.endswith("uncorrected") or key.startswith("remapped_rows"):
                out.append(Finding("fail", "logs.counters", f"GPU{delta['index']} {key} +{increase}"))
            elif nvlink_over or key.endswith("corrected"):
                out.append(Finding("warn", "logs.counters", f"GPU{delta['index']} {key} +{increase}"))
        if delta["hw_throttle"]:
            out.append(Finding("warn", "logs.throttle", f"GPU{delta['index']} {','.join(delta['hw_throttle'])}"))
    if data["nccl"]["count"]:
        out.append(
            Finding(
                "fail",
                "logs.nccl",
                f"{data['nccl']['count']} NCCL log matches, e.g. {data['nccl']['matches'][0][-160:]}",
            )
        )
    for source, result in data["kernel"].items():
        if result.get("matches"):
            out.append(
                Finding(
                    "fail", f"logs.{source}", f"{len(result['matches'])} matches, e.g. {result['matches'][0][-160:]}"
                )
            )
    return out


def check_homogeneity(
    sysinfos: dict[str, dict[str, Any]],
    *,
    fail_keys: list[str],
    warn_keys: list[str],
) -> dict[str, list[Finding]]:
    """fail_keys and warn_keys are key paths into sysinfo that should match on every node.

    Nodes outside the majority value are flagged; with no value on more than half the nodes, all of them are."""
    out: dict[str, list[Finding]] = defaultdict(list)
    severity: Severity
    for severity, keys in (("fail", fail_keys), ("warn", warn_keys)):
        for key in keys:
            values = {host: json.dumps(resolve(data, key), default=str) for host, data in sysinfos.items()}
            if len(set(values.values())) < 2:
                continue
            most_common, count = Counter(values.values()).most_common(1)[0]
            if count * 2 <= len(values):
                for host, value in values.items():
                    out[host].append(Finding(severity, "homogeneity", f"{key}={value} (no majority)"))
                continue
            for host, value in values.items():
                if value != most_common:
                    out[host].append(Finding(severity, "homogeneity", f"{key}={value} (majority {most_common})"))
    return out


def check_scratch_median(gbps_of: dict[str, float], cfg: ScratchRead) -> tuple[dict[str, list[Finding]], float | None]:
    """Shared-filesystem throughput depends on load, so compare nodes against each other too."""
    out: dict[str, list[Finding]] = defaultdict(list)
    median = _median(list(gbps_of.values()))
    floor = (median or 0) * cfg.min_frac_of_median
    for host, gbps in gbps_of.items():
        if gbps < floor:
            out[host].append(Finding(cfg.severity, "scratch.read", f"read {gbps} < {floor:.2f} GB/s (frac of median)"))
    return out, median


def check_nccl(groups: list[dict[str, Any]], cfg: Nccl) -> tuple[dict[str, list[Finding]], dict]:
    out: dict[str, list[Finding]] = defaultdict(list)
    tests = [test for group in groups for test in group["tests"]]
    by_kind = defaultdict(list)
    for test in tests:
        by_kind[test["kind"]].append(test)
    medians = {
        kind: _median([test["peak_busbw"] for test in kind_tests if ok(test)]) for kind, kind_tests in by_kind.items()
    }

    for kind, minimum in (("intra", cfg.intra_min_busbw_GBps), ("loopback", cfg.loopback_min_busbw_GBps)):
        for test in by_kind[kind]:
            if not ok(test):
                out[test["nodes"][0]].append(Finding("fail", f"nccl.{kind}", f"{test['status']} ({test['log']})"))
            elif test["peak_busbw"] < minimum:
                out[test["nodes"][0]].append(Finding("fail", f"nccl.{kind}", f"busbw {test['peak_busbw']} < {minimum}"))

    # A bad node makes both of its ring pairs slow; its healthy neighbours see only one. A node whose every pair is
    # bad fails too: with two nodes there is one pair, and nothing tells the two apart.
    pair_floor = max(cfg.pair_min_busbw_GBps, (medians.get("pair") or 0) * cfg.pair_min_frac_of_median)
    slow: dict[str, list[str]] = defaultdict(list)
    pairs_of: Counter[str] = Counter()
    for test in by_kind["pair"]:
        pairs_of.update(test["nodes"])
        if not ok(test) or test["peak_busbw"] < pair_floor:
            for node in test["nodes"]:
                other = next(peer for peer in test["nodes"] if peer != node)
                slow[node].append(f"{other}: {test['status']} {test['peak_busbw']}")
    for node, partners in slow.items():
        severity: Severity = "fail" if len(partners) >= 2 or len(partners) == pairs_of[node] else "warn"
        out[node].append(
            Finding(
                severity,
                "nccl.pair",
                f"{len(partners)}/{pairs_of[node]} slow pairs (< {pair_floor:.1f}): {'; '.join(partners)}",
            )
        )

    group_floor = max(cfg.group_min_busbw_GBps, (medians.get("group") or 0) * cfg.group_min_frac_of_median)
    for test in by_kind["group"]:
        if not ok(test) or test["peak_busbw"] < group_floor:
            # A hung group with no node already failing for it can't be blamed on one, so every node in it fails
            blamed = any(finding.severity == "fail" for node in test["nodes"] for finding in out.get(node, []))
            severity = "fail" if test["status"] == Status.TIMEOUT and not blamed else "warn"
            for node in test["nodes"]:
                out[node].append(
                    Finding(
                        severity,
                        "nccl.group",
                        f"{len(test['nodes'])}-node all_reduce {test['status']} "
                        f"busbw {test['peak_busbw']} (floor {group_floor:.1f})",
                    )
                )
    return out, medians


def evaluate(run: Path, cfg: Config, switch_of: dict[str, str]) -> dict[str, Any]:
    """switch_of maps every node the run targeted to its leaf switch."""
    layout = Paths(root=run)
    jobs = read_json(layout.jobs) or []
    job_of = {node: job for job in jobs for node in job["nodes"]}
    findings: dict[str, list[Finding]] = defaultdict(list)
    sysinfos = {}
    scratch = {}
    step_seconds: dict[str, dict[str, float]] = {}

    hosts = {path.stem for path in layout.nodes.glob("*.json")} | set(switch_of) | set(job_of)
    for host in sorted(hosts):
        node = read_json(layout.node(host))
        if node is None:
            job = job_of.get(host, {})
            findings[host].append(Finding("fail", "no_report", f"job {job.get('job_id')} {job.get('state', '?')}"))
            continue
        step_seconds[host] = {name: node[name]["elapsed_s"] for name in STEPS if node.get(name)}
        if data := _step(node, "sysinfo", findings[host]):
            sysinfos[host] = data
            findings[host] += check_sysinfo(data, cfg.gpu, cfg.nic)
        if data := _step(node, "scratch", findings[host]):
            findings[host] += check_scratch_read(data, cfg.scratch.read)
            if ok(data):
                scratch[host] = data["gbps"]
        if data := _step(node, "local_tmp", findings[host]):
            findings[host] += check_local_tmp(data, cfg.local_tmp)
        if data := _step(node, "dcgm", findings[host]):
            findings[host] += check_dcgm(data, cfg.dcgm)
        if node.get("logs") is None:
            findings[host].append(Finding("warn", "logs", "finalize did not run"))
        elif data := _step(node, "logs", findings[host]):
            findings[host] += check_logs(data, cfg.logs)

    # Checks across nodes
    scratch_findings, scratch_median = check_scratch_median(scratch, cfg.scratch.read)
    groups = [group for path in sorted(layout.groups.glob("*.json")) if (group := read_json(path))]
    nccl_findings, nccl_medians = check_nccl(groups, cfg.nccl)
    homogeneity_findings = check_homogeneity(sysinfos, fail_keys=cfg.homogeneity.fail, warn_keys=cfg.homogeneity.warn)
    for by_host in (scratch_findings, homogeneity_findings, nccl_findings):
        for host, fs in by_host.items():
            findings[host] += fs
    # GBps: gigabytes per second, as nccl-tests reports busbw
    medians = {"scratch_read_GBps": scratch_median}
    medians |= {f"nccl_{kind}_busbw_GBps": median for kind, median in nccl_medians.items()}

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
    switch_of: dict[str, str],
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
