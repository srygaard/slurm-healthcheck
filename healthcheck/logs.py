"""Step 4: error signals raised while the tests ran.

Kernel logs need root on most clusters, so the main signals are the NVML Xid watcher,
GPU counter deltas and the NCCL debug logs, which a normal user can always read.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from healthcheck import xid
from healthcheck.check import Availability, Finding, HealthCheckBase, Report, per_host
from healthcheck.config import Logs, Paths
from healthcheck.sysinfo import gpu
from healthcheck.util.cmd import run
from healthcheck.util.file import Json, JsonDict
from healthcheck.util.hostlist import ByHost
from healthcheck.util.status import Status


def _matches(lines: list[str], patterns: list[str], limit: int = 50) -> list[str]:
    regex = re.compile("|".join(map(re.escape, patterns)))
    return [line for line in lines if regex.search(line)][:limit]


def kernel(patterns: list[str], since: float) -> JsonDict:
    out: dict[str, Json] = {}
    dmesg = run(["dmesg", "--time-format", "iso", "--level", "emerg,alert,crit,err,warn"], timeout=15)
    if dmesg.returncode:
        out["dmesg"] = {"status": Status.UNAVAILABLE, "error": dmesg.stderr.strip()[:200]}
    else:
        recent = []
        for line in dmesg.stdout.splitlines():
            try:
                stamp = datetime.fromisoformat(line.split()[0].replace(",", "."))
            except (ValueError, IndexError):
                continue
            if stamp.timestamp() >= since:
                recent.append(line)
        out["dmesg"] = {"status": Status.OK, "matches": _matches(recent, patterns)}

    journal = run(["journalctl", "-k", "-p", "warning", "--since", f"@{int(since)}", "--no-pager", "-q"], timeout=15)
    if journal.returncode or "insufficient permissions" in journal.stderr + journal.stdout:
        out["journal"] = {"status": Status.UNAVAILABLE, "error": (journal.stderr or journal.stdout).strip()[:200]}
    else:
        out["journal"] = {"status": Status.OK, "matches": _matches(journal.stdout.splitlines(), patterns)}
    return out


def _rank_log(host: str) -> re.Pattern[str]:
    return re.compile(rf"[^.]*\.{re.escape(host)}(\.[\w-]+)*\.\d+\.log")


def nccl_logs(log_dir: Path, host: str, patterns: list[str]) -> JsonDict:
    """
    The host's ranks' debug logs, <test label>.<host>.<pid>.log: a pair's label names the other host too, and %h may
    add the host's domain.

    >>> name = _rank_log("n1")
    >>> [bool(name.fullmatch(log)) for log in ["pair-n1-n2.n1.7.log", "pair-n1-n2.n2.7.log", "intra-n1.n1.ex.org.7.log"]]
    [True, False, True]
    >>> bool(name.fullmatch("intra-rack-n1.rack-n1.7.log"))
    False
    """
    name = _rank_log(host)
    files = sorted(path for path in log_dir.glob("*.log") if name.fullmatch(path.name))
    lines = [line for log in files for line in log.read_text(errors="replace").splitlines()]
    matches = _matches(lines, patterns, limit=10_000)
    return {"files": len(files), "count": len(matches), "matches": matches[:30]}


def counter_deltas(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Increases in any error counter, plus hardware throttling seen at the end."""

    def flatten(value: Any, prefix: str = "") -> dict[str, int]:
        if isinstance(value, dict):
            return {
                path: count for key, sub in value.items() for path, count in flatten(sub, f"{prefix}{key}.").items()
            }
        return {prefix.rstrip("."): value} if isinstance(value, int) and not isinstance(value, bool) else {}

    deltas: list[JsonDict] = []
    for gpu_before, gpu_after in zip(before, after):
        flat_before = flatten({key: gpu_before[key] for key in gpu_before if key != "index"})
        flat_after = flatten({key: gpu_after[key] for key in gpu_after if key != "index"})
        changed = {
            key: flat_after[key] - flat_before.get(key, 0)
            for key in flat_after
            if flat_after[key] > flat_before.get(key, 0)
        }
        if changed or gpu_after["hw_throttle"]:
            deltas.append({"index": gpu_after["index"], "increased": changed, "hw_throttle": gpu_after["hw_throttle"]})
    return deltas


class HealthCheck(HealthCheckBase):
    @classmethod
    def missing(cls, name: str) -> Finding:
        return Finding("warn", name, "finalize did not run")

    def __init__(
        self,
        cfg: Logs,
        run: Paths,
        host: str,
        since: float,
        before: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        self.cfg = cfg
        self.run = run
        self.host = host
        self.since = since
        self.before = before

    @property
    def enabled(self) -> bool:
        return True

    def available(self) -> Availability:
        return Availability(True)

    def collect(self) -> JsonDict:
        after = [snapshot.to_json() for snapshot in gpu.states()]
        return {
            "kernel": kernel(self.cfg.kernel_patterns, self.since),
            "nccl": nccl_logs(self.run.nccl_logs, self.host, self.cfg.nccl_fail_patterns),
            "xid": xid.events(self.run.xid(self.host), self.cfg.xid_ignore),
            "counters": after,
            "counter_deltas": counter_deltas(self.before, after) if self.before else None,
        }

    @staticmethod
    def report(results: ByHost[JsonDict], cfg: Logs, name: str) -> Report:
        return Report(per_host(results, cfg, name, _findings), {})


def _findings(data: JsonDict, cfg: Logs) -> Iterator[Finding]:
    for event in data["xid"].get("events", []):
        yield Finding("fail", "logs.xid", f"GPU{event['gpu']} Xid {event['xid']} at {event['time']}")
    for delta in data.get("counter_deltas") or []:
        for key, increase in delta["increased"].items():
            if key.startswith("ecc_aggregate"):
                continue  # rises with ecc_volatile, which is reported
            nvlink_over = key.startswith("nvlink_errors") and increase > cfg.max_nvlink_error_delta
            if key.endswith("uncorrected") or key.startswith("remapped_rows"):
                yield Finding("fail", "logs.counters", f"GPU{delta['index']} {key} +{increase}")
            elif nvlink_over or key.endswith("corrected"):
                yield Finding("warn", "logs.counters", f"GPU{delta['index']} {key} +{increase}")
        if delta["hw_throttle"]:
            yield Finding("warn", "logs.throttle", f"GPU{delta['index']} {','.join(delta['hw_throttle'])}")
    if data["nccl"]["count"]:
        yield Finding(
            "fail",
            "logs.nccl",
            f"{data['nccl']['count']} NCCL log matches, e.g. {data['nccl']['matches'][0][-160:]}",
        )
    for source, result in data["kernel"].items():
        yield from _kernel_findings(source, result.get("matches") or [], cfg.kernel_warn)


def _kernel_findings(source: str, matches: list[str], warn: list[str]) -> Iterator[Finding]:
    """A line matching kernel_warn warns. Any other matched line fails the node.

    >>> list(_kernel_findings("dmesg", ["Failed to send inband data: 0", "NVRM: Xid 79"], ["inband data: 0"]))
    [Finding(severity='warn', check='logs.dmesg', detail='1 matches, e.g. Failed to send inband data: 0'), \
Finding(severity='fail', check='logs.dmesg', detail='1 matches, e.g. NVRM: Xid 79')]
    """
    failed: list[str] = []
    warned: dict[str, list[str]] = {}
    for line in matches:
        pattern = next((item for item in warn if item in line), None)
        if pattern is None:
            failed.append(line)
        else:
            warned.setdefault(pattern, []).append(line)
    for lines in warned.values():
        yield Finding("warn", f"logs.{source}", f"{len(lines)} matches, e.g. {lines[0][-160:]}")
    if failed:
        yield Finding("fail", f"logs.{source}", f"{len(failed)} matches, e.g. {failed[0][-160:]}")
