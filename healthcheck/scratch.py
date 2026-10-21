"""Step 2a: $SCRATCH read throughput. `python -m healthcheck.scratch` writes the read-test files (build job)."""

import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from healthcheck.config import Scratch, cfg
from healthcheck.util.file import JsonDict
from healthcheck.util.io_throughput import read_throughput


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


def collect(base: Path, cfg: Scratch, host: str) -> JsonDict:
    """Read one of the build job's files; spread nodes across files to avoid one hot OST set."""
    path = file_path(base, zlib.crc32(host.encode()) % cfg.files)
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return {
        "status": "ok",
        "path": str(path),
        **read_throughput(path, float("inf"), cfg.read.max_s, cfg.read.block_mib * 2**20, loop=True),
    }


def main() -> int:
    size = int(cfg.scratch.file_gib * 2**30)
    block = cfg.scratch.write_block_mib * 2**20
    paths = [file_path(cfg.paths.base, index) for index in range(cfg.scratch.files)]
    print(f"{len(paths)} x {cfg.scratch.file_gib} GiB read-test files in {paths[0].parent}", flush=True)
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        written = sum(pool.map(write_random, paths, [size] * len(paths), [block] * len(paths)))
    print(f"wrote {written}, {len(paths) - written} already there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
