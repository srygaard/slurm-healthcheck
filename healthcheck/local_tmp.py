"""Step 2b: local tmp read/write throughput, SMART and free space."""

import json
import os
import shutil
from pathlib import Path

from healthcheck.config import LocalTmp
from healthcheck.util.cmd import run
from healthcheck.util.file import Json, JsonDict
from healthcheck.util.io_throughput import read_throughput, write_throughput


def rw(directory: Path, host: str, gib: float, max_s: float, block_mib: int) -> JsonDict:
    path = directory / f".healthcheck-{host}-{os.getpid()}"
    try:
        block = block_mib * 2**20
        write = write_throughput(path, int(gib * 2**30), max_s, block)
        read = read_throughput(path, write["bytes"], max_s, block)
    finally:
        path.unlink(missing_ok=True)
    return {"write": write, "read": read}


def smart() -> JsonDict:
    """SMART health per local disk; needs root on most systems, so unavailable is expected."""
    lsblk = run(["lsblk", "-J", "-d", "-o", "NAME,TYPE,TRAN,MODEL,SIZE"], timeout=10)
    if lsblk.returncode:
        return {"status": "unavailable", "error": lsblk.stderr.strip()}
    disks = [device for device in json.loads(lsblk.stdout)["blockdevices"] if device["type"] == "disk"]
    results: dict[str, JsonDict] = {}
    for disk in disks:
        out = run(["smartctl", "-H", "-j", f"/dev/{disk['name']}"], timeout=15)
        try:
            report = json.loads(out.stdout)
        except json.JSONDecodeError:
            results[disk["name"]] = {"status": "unavailable", "error": out.stderr.strip()[:200]}
            continue
        passed = report.get("smart_status", {}).get("passed")
        messages = [message.get("string") for message in report.get("smartctl", {}).get("messages", [])]
        results[disk["name"]] = {
            "status": "unavailable" if passed is None else "ok",
            "passed": passed,
            "model": disk.get("model"),
            "messages": messages,
        }
    status = "ok" if any(result["status"] == "ok" for result in results.values()) else "unavailable"
    return {"status": status, "disks": results}


def collect(cfg: LocalTmp, host: str) -> JsonDict:
    """Free space for every path; read/write only for the first path on each filesystem."""
    expanded = [os.path.expandvars(path) for path in cfg.paths]
    # An empty expansion would mean "."; dict.fromkeys drops paths that resolve to the same directory, in order
    directories = dict.fromkeys(Path(path).resolve() for path in expanded if path and Path(path).is_dir())
    first_on_device: dict[int, Path] = {}
    paths = {}
    for directory in directories:
        usage = shutil.disk_usage(directory)
        entry: dict[str, Json] = {"free_gib": round(usage.free / 2**30, 1), "total_gib": round(usage.total / 2**30, 1)}
        device = directory.stat().st_dev
        if device in first_on_device:
            entry["same_device_as"] = str(first_on_device[device])
        else:
            first_on_device[device] = directory
            try:
                entry |= rw(directory, host, cfg.rw.gib, cfg.rw.max_s, cfg.rw.block_mib)
            except OSError as error:
                entry["error"] = str(error)
        paths[str(directory)] = entry
    return {"paths": paths, "smart": smart()}
