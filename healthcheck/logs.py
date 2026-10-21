"""Step 4: error signals raised while the tests ran.

Kernel logs need root on most clusters, so the main signals are the NVML Xid watcher,
GPU counter deltas and the NCCL debug logs, which a normal user can always read.
"""

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from healthcheck.config import Logs, Paths
from healthcheck.sysinfo import gpu
from healthcheck.util.cmd import run
from healthcheck.util.file import Json, JsonDict


def _matches(lines: list[str], patterns: list[str], limit: int = 50) -> list[str]:
    regex = re.compile("|".join(map(re.escape, patterns)))
    return [line for line in lines if regex.search(line)][:limit]


def kernel(patterns: list[str], since: float) -> JsonDict:
    out: dict[str, Json] = {}
    dmesg = run(["dmesg", "--time-format", "iso", "--level", "emerg,alert,crit,err,warn"], timeout=15)
    if dmesg.returncode:
        out["dmesg"] = {"status": "unavailable", "error": dmesg.stderr.strip()[:200]}
    else:
        recent = []
        for line in dmesg.stdout.splitlines():
            try:
                stamp = datetime.fromisoformat(line.split()[0].replace(",", "."))
            except (ValueError, IndexError):
                continue
            if stamp.timestamp() >= since:
                recent.append(line)
        out["dmesg"] = {"status": "ok", "matches": _matches(recent, patterns)}

    journal = run(["journalctl", "-k", "-p", "warning", "--since", f"@{int(since)}", "--no-pager", "-q"], timeout=15)
    if journal.returncode or "insufficient permissions" in journal.stderr + journal.stdout:
        out["journal"] = {"status": "unavailable", "error": (journal.stderr or journal.stdout).strip()[:200]}
    else:
        out["journal"] = {"status": "ok", "matches": _matches(journal.stdout.splitlines(), patterns)}
    return out


def nccl_logs(log_dir: Path, host: str, patterns: list[str]) -> JsonDict:
    files = sorted(log_dir.glob(f"{host}*.log"))
    lines = [line for log in files for line in log.read_text(errors="replace").splitlines()]
    matches = _matches(lines, patterns, limit=10_000)
    return {"files": len(files), "count": len(matches), "matches": matches[:30]}


def xid_events(path: Path, ignore: list[int]) -> JsonDict:
    try:
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return {"status": "unavailable", "error": "watcher wrote nothing"}
    xids = [event for event in events if event.get("event") == "xid"]
    errors = [event for event in events if event.get("event") == "error"]
    return {
        "status": "ok" if any(event.get("event") == "started" for event in events) else "unavailable",
        "events": [xid for xid in xids if xid["xid"] not in ignore],
        "ignored": [xid for xid in xids if xid["xid"] in ignore],
        "errors": errors,
    }


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


def collect(
    cfg: Logs,
    run: Paths,
    host: str,
    since: float,
    before: Sequence[Mapping[str, Any]] | None,
) -> JsonDict:
    after = gpu.counters()
    return {
        "kernel": kernel(cfg.kernel_patterns, since),
        "nccl": nccl_logs(run.nccl_logs, host, cfg.nccl_fail_patterns),
        "xid": xid_events(run.xid(host), cfg.xid_ignore),
        "counters": after,
        "counter_deltas": counter_deltas(before, after) if before else None,
    }
