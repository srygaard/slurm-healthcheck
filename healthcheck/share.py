"""Step 2a: shared filesystem read throughput, $SCRATCH and $PROJECT, one step each.

`python -m healthcheck.share` writes the read-test files of the enabled ones (`healthcheck.py setup`).
"""

import os
import zlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from healthcheck.check import Availability, Finding, HealthCheckBase, Report, below_median, per_host
from healthcheck.config import Scratch, cfg
from healthcheck.util.file import JsonDict
from healthcheck.util.hostlist import ByHost
from healthcheck.util.io_throughput import read_throughput
from healthcheck.util.status import Status, ok


def file_path(base: Path, index: int) -> Path:
    return base / "diskprobe" / f"f{index}"


def write_random(path: Path, size: int, block: int) -> bool:
    """Random data, so compression anywhere on the path can't flatter the read rate; False if already written."""
    if path.is_file() and path.stat().st_size >= size:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "wb") as out:
        out.writelines(os.urandom(block) for _ in range(-(-size // block)))
    tmp.replace(path)
    return True


class HealthCheck(HealthCheckBase):
    def __init__(self, base: Path, cfg: Scratch, host: str) -> None:
        self.base = base
        self.cfg = cfg
        self.host = host

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def available(self) -> Availability:
        if self.base.is_dir():
            return Availability(True)
        return Availability(False, f"{self.base} is not a directory")

    def collect(self) -> JsonDict:
        """Read one of the files written by setup; spread nodes across files to avoid one hot OST set."""
        path = file_path(self.base, zlib.crc32(self.host.encode()) % self.cfg.files)
        if not path.is_file():
            return {"status": Status.MISSING, "path": str(path)}
        return {
            "status": Status.OK,
            "path": str(path),
            **read_throughput(path, float("inf"), self.cfg.read_max_s, self.cfg.read_block_mib * 2**20, loop=True),
        }

    @staticmethod
    def report(results: ByHost[JsonDict], cfg: Scratch, name: str) -> Report:
        """name is the step, scratch or project: both run this check and read their own thresholds."""
        findings = per_host(results, cfg, name, lambda data, section: _findings(data, section, name))
        gbps = {
            host: data["gbps"]
            for host, data in results.items()
            if ok(data)
        }
        if not gbps:
            return Report(findings, {})
        mid, floor, slow = below_median(gbps, cfg.min_frac_of_median)
        for host in slow:
            detail = f"read {gbps[host]} < {floor:.2f} GB/s (frac of median)"
            findings.setdefault(host, []).append(Finding(cfg.severity, f"{name}.read", detail))
        return Report(findings, {f"{name}_read_GBps": mid})


def _findings(data: JsonDict, cfg: Scratch, name: str) -> Iterator[Finding]:
    """One node's read of a shared filesystem. name labels the findings."""
    if not ok(data):
        yield Finding("warn", f"{name}.read", f"{data['status']}: {data['path']} (run setup)")
        return
    if data["gbps"] < cfg.min_GBps:
        yield Finding(cfg.severity, f"{name}.read", f"read {data['gbps']} < {cfg.min_GBps} GB/s")


def write_files(base: Path, cfg: Scratch) -> None:
    size = int(cfg.file_gib * 2**30)
    block = cfg.write_block_mib * 2**20
    paths = [file_path(base, index) for index in range(cfg.files)]
    print(f"{len(paths)} x {cfg.file_gib} GiB read-test files in {paths[0].parent}")
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        written = sum(pool.map(write_random, paths, [size] * len(paths), [block] * len(paths)))
    print(f"wrote {written}, {len(paths) - written} already there")


def main() -> int:
    if cfg.scratch.enabled:
        write_files(cfg.paths.base, cfg.scratch)
    if cfg.project.enabled:
        write_files(cfg.paths.project, cfg.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
