"""Setup probe on one GPU node: how to reach DCGM and which srun MPI plugin works, saved to <base>/env.json."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from healthcheck import dcgm, nccl
from healthcheck.config import cfg
from healthcheck.util.file import write_json


def main() -> int:
    base = cfg.paths.base
    env: dict[str, Any] = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "probe_job": os.environ.get("SLURM_JOB_ID"),
        "probe_node": cfg.host.name,
        "nccl_tests": str(Path(os.environ["HC_NCCL_TESTS"]) / "build"),
        "dcgm": "none",
        "attempts": [],
    }

    # Prefer the host's running hostengine, then one we start ourselves
    dcgmi = shutil.which("dcgmi")
    hostengine = shutil.which("nv-hostengine")
    print(f"host dcgmi: {dcgmi or 'none'}, nv-hostengine: {hostengine or 'none'}", flush=True)
    modes = [mode for mode, available in [("host", dcgmi), ("host-self", dcgmi and hostengine)] if available]
    for mode in modes:
        try:
            result = dcgm.diag(
                mode,
                suite="1",
                port=cfg.dcgm.port,
                timeout=180,
                log_dir=base / "results" / str(env["probe_job"]) / f"dcgm-{mode}",
                parameters=";".join(cfg.dcgm.parameters),
            )
        except dcgm.DiagError as error:  # record it, then try the next mode
            env["attempts"].append(
                {
                    "dcgm": mode,
                    "error": str(error),
                    "returncode": error.out.returncode,
                    "stdout": error.out.stdout[-1500:],
                    "stderr": error.out.stderr[-500:],
                }
            )
            continue
        statuses = {name: test["status"] for name, test in result["tests"].items()}
        env["attempts"].append({"dcgm": mode, "returncode": result["returncode"], "tests": statuses})
        # Parsed output alone isn't enough: a refused run still reports its failed environment check
        if "Pass" in statuses.values():
            env["dcgm"] = mode
            break

    binary = Path(env["nccl_tests"]) / cfg.nccl.binary
    for mpi in ("pmix", "pmi2"):
        launch = nccl.Launch(mpi, binary, ["--minbytes=8M", "--maxbytes=8M"], cfg.gpu.count)
        [result] = nccl.run_round(
            [nccl.Run("probe", (cfg.host.name,))], launch, base / "results" / str(env["probe_job"]) / mpi, timeout=120
        )
        env["attempts"].append({"mpi": mpi, **result})
        if result["status"] == "ok":
            env["mpi"] = mpi
            break

    write_json(cfg.paths.env, env)
    print(f"dcgm: {env['dcgm']}  mpi: {env.get('mpi', 'none')}  ->  {cfg.paths.env}")
    for attempt in env["attempts"]:
        print(f"  {attempt}")
    return 0 if env.get("mpi") else 1


if __name__ == "__main__":
    raise SystemExit(main())
