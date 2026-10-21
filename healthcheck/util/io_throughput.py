"""Sequential file read and write throughput, bypassing the page cache where the filesystem allows."""

import errno
import mmap
import os
import time
from pathlib import Path
from typing import Any


def _open(path: Path, flags: int) -> tuple[int, bool]:
    """Open with O_DIRECT when the filesystem allows it (tmpfs and older ZFS don't)."""
    try:
        return os.open(path, flags | os.O_DIRECT, 0o600), True
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
        return os.open(path, flags, 0o600), False


def _throughput(done: int, start: float, direct: bool) -> dict[str, Any]:
    elapsed = time.monotonic() - start
    return {"gbps": round(done / elapsed / 1e9, 3), "bytes": done, "seconds": round(elapsed, 2), "direct": direct}


def read_throughput(path: Path, max_bytes: float, max_s: float, block: int, loop: bool = False) -> dict[str, Any]:
    """Read sequentially; with loop, start over at end of file so the window lasts max_s whatever the size."""
    fd, direct = _open(path, os.O_RDONLY)
    done = 0
    start = time.monotonic()
    try:
        if not direct:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # best effort to skip the page cache
        with mmap.mmap(-1, block) as buf:  # page-aligned, as O_DIRECT requires
            while done < max_bytes and time.monotonic() - start < max_s:
                if not (read := os.readv(fd, [buf])):
                    if not loop or not done:
                        break
                    os.lseek(fd, 0, os.SEEK_SET)
                    continue
                done += read
    finally:
        os.close(fd)
    return _throughput(done, start, direct)


def write_throughput(path: Path, max_bytes: int, max_s: float, block: int) -> dict[str, Any]:
    fd, direct = _open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        with mmap.mmap(-1, block) as buf:  # page-aligned, as O_DIRECT requires
            buf.write(os.urandom(block))  # incompressible, so compressing filesystems can't flatter the result
            done = 0
            start = time.monotonic()
            while done < max_bytes and time.monotonic() - start < max_s:
                done += os.write(fd, buf)
        os.fsync(fd)
    finally:
        os.close(fd)
    return _throughput(done, start, direct)
