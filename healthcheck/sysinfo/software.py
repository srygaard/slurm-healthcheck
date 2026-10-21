"""OS release and the driver, fabric and module versions NCCL depends on."""

import os
import sys
from pathlib import Path

from healthcheck.util.cmd import run
from healthcheck.util.file import JsonDict, read_sys


def os_release() -> str | None:
    for line in (read_sys("/etc/os-release") or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return None


def collect() -> JsonDict:
    ofed = run(["ofed_info", "-s"], timeout=10)
    return {
        "ofed": ofed.stdout.strip().rstrip(":") if ofed.returncode == 0 else read_sys("/sys/module/mlx5_core/version"),
        "nvidia_module": read_sys("/sys/module/nvidia/version"),
        "gdrdrv": Path("/sys/module/gdrdrv").exists(),
        "nccl_module": os.environ.get("EBVERSIONNCCL"),
        "cuda_module": os.environ.get("EBVERSIONCUDACORE") or os.environ.get("EBVERSIONCUDA"),
        "python": sys.version.split()[0],
    }
