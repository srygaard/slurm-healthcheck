"""Step 3c: nccl-tests across the job's nodes, launched as concurrent srun steps from the batch step.

Rounds, each a set of concurrent steps on disjoint nodes:
  intra     every node alone: NVLink/NVSwitch
  loopback  every node alone with P2P/SHM off, so traffic goes through its NICs
  pairs     ring neighbours (i, i+1): a bad NIC shows up in both of a node's pairs
  group     all nodes at once
"""

import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from healthcheck.config import cfg
from healthcheck.util.file import Json, read_json, write_json
from healthcheck.util.hostlist import Hostlist
from healthcheck.util.status import Status
from healthcheck.util.switches import node_switches


class Run(NamedTuple):
    kind: str
    nodes: tuple[str, ...]
    env: dict[str, str] = {}

    @property
    def label(self) -> str:
        return f"{self.kind}-{self.nodes[0]}" + (f"-{self.nodes[-1]}" if len(self.nodes) > 1 else "")


def parse(text: str) -> dict[str, Any]:
    """Size table, peak out-of-place busbw and #wrong from nccl-tests stdout."""
    columns: list[str] = []
    rows = []
    for line in text.splitlines():
        tokens = line.split()
        if tokens[:2] == ["#", "size"]:
            columns = tokens[1:]
        elif columns and tokens and tokens[0].isdigit() and len(tokens) == len(columns):
            busbw = columns.index("busbw")  # first occurrence is out-of-place
            wrong = columns.index("#wrong")
            rows.append(
                {
                    "size": int(tokens[0]),
                    "busbw": float(tokens[busbw]),
                    "wrong": int(tokens[wrong]) if tokens[wrong].isdigit() else 0,
                }
            )
    avg = re.search(r"# Avg bus bandwidth\s*:\s*([\d.]+)", text)
    return {
        "rows": rows,
        "peak_busbw": max((row["busbw"] for row in rows), default=None),
        "avg_busbw": float(avg.group(1)) if avg else None,
        "wrong": sum(row["wrong"] for row in rows),
    }


def pair_rounds(nodes: list[str]) -> list[list[tuple[str, str]]]:
    """
    Ring pairs (i, i+1) split into rounds where no node appears twice.

    >>> pair_rounds(["a", "b", "c", "d"])
    [[('a', 'b'), ('c', 'd')], [('b', 'c'), ('d', 'a')]]
    >>> pair_rounds(["a", "b", "c"])
    [[('a', 'b')], [('b', 'c')], [('c', 'a')]]
    """
    count = len(nodes)
    if count < 2:
        return []
    if count == 2:
        return [[(nodes[0], nodes[1])]]
    rounds: list[list[tuple[str, str]]] = []
    for index in range(count):
        pair = (nodes[index], nodes[(index + 1) % count])
        for round_ in rounds:
            if not {node for other in round_ for node in other} & set(pair):
                round_.append(pair)
                break
        else:
            rounds.append([pair])
    return rounds


class Launch(NamedTuple):
    """How every test of a run is started: the same binary, arguments and ranks per node."""

    mpi: str
    binary: Path
    args: list[str]
    ranks_per_node: int

    @property
    def cpus_per_task(self) -> int:
        """The batch step's CPUs shared evenly among the ranks on a node."""
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", self.ranks_per_node)) // self.ranks_per_node)

    def command(self, test: Run) -> list[str]:
        # GPUs aren't bound per task: nccl-tests picks its device from the local rank.
        return [
            "srun",
            "--exact",
            "--kill-on-bad-exit=1",
            f"--mpi={self.mpi}",
            f"--nodes={len(test.nodes)}",
            f"--ntasks={len(test.nodes) * self.ranks_per_node}",
            f"--ntasks-per-node={self.ranks_per_node}",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--nodelist={Hostlist(test.nodes)}",
            str(self.binary),
            *self.args,
        ]


def run_round(
    tests: list[Run],
    launch: Launch,
    out_dir: Path,
    timeout: float | None = None,
    debug_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run tests concurrently; a straggler past the timeout, if given, is terminated and reported as such.

    With debug_dir, each rank writes NCCL's debug log to <debug_dir>/<label>.<host>.<pid>.log, and the empty ones of
    tests that finished are removed after it: at NCCL_DEBUG=WARN a healthy rank writes nothing, rank 0 only the
    version. A timed-out test's are all kept, since its remote ranks may outlive srun and still be writing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    start = time.monotonic()
    for test in tests:
        log = out_dir / f"{test.label}.txt"
        env = os.environ | test.env
        if debug_dir is not None:
            env["NCCL_DEBUG_FILE"] = str(debug_dir / f"{test.label}.%h.%p.log")
        with open(log, "w") as output:
            procs.append(
                (
                    test,
                    log,
                    subprocess.Popen(launch.command(test), stdout=output, stderr=subprocess.STDOUT, env=env),
                )
            )
    # Poll rather than wait on each in turn, so each test's elapsed time is its own
    ended: dict[int, float] = {}
    while True:
        now = time.monotonic()
        for index, (_, _, proc) in enumerate(procs):
            if index not in ended and proc.poll() is not None:
                ended[index] = now
        if len(ended) == len(procs) or (timeout is not None and now - start >= timeout):
            break
        time.sleep(0.2)
    results = []
    for index, (test, log, proc) in enumerate(procs):
        status = Status.OK if index in ended else Status.TIMEOUT
        if status == Status.TIMEOUT:
            proc.send_signal(signal.SIGTERM)  # srun cancels the step
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        parsed = parse(log.read_text(errors="replace"))
        if status == Status.OK and (proc.returncode or parsed["peak_busbw"] is None):
            status = Status.ERROR
        elif status == Status.OK and parsed["wrong"]:
            status = Status.WRONG
        results.append(
            {
                "kind": test.kind,
                "nodes": list(test.nodes),
                "status": status,
                "returncode": proc.returncode,
                "elapsed_s": round(ended.get(index, now) - start, 1),
                "peak_busbw": parsed["peak_busbw"],
                "avg_busbw": parsed["avg_busbw"],
                "wrong": parsed["wrong"],
                "log": str(log),
            }
        )
    if debug_dir is not None:
        # Only this round's finished tests: the run's other jobs share debug_dir, and their ranks may still be writing
        for index, (test, _, _) in enumerate(procs):
            for path in debug_dir.glob(f"{test.label}.*.log") if index in ended else ():
                if path.stat().st_size == 0:
                    path.unlink(missing_ok=True)
    return results


def plan(
    nodes: list[str],
    loopback_env: dict[str, str],
    switch_of: Mapping[str, str],
) -> list[list[Run]]:
    """
    Rounds of concurrent tests; each kind takes about the same time at any node count.

    The group test runs once per leaf switch with more than two of the nodes, all at once: a group across the whole
    job would slow down with its size, and the pairs already name a bad node. Nodes with no known switch (a cluster
    with topology/flat, like Trillium's non-blocking fabric) get no group test, since it would span the whole job.

    >>> [[test.label for test in tests] for tests in plan(["a", "b", "c", "d"], {}, {"a": "s1", "b": "s1"})][2:]
    [['pair-a-b', 'pair-c-d'], ['pair-b-c', 'pair-d-a']]
    >>> plan(list("abcdef"), {}, dict.fromkeys("abc", "s1") | dict.fromkeys("def", "s2"))[-1]
    [Run(kind='group', nodes=('a', 'b', 'c'), env={}), Run(kind='group', nodes=('d', 'e', 'f'), env={})]
    >>> [test.kind for test in plan(list("abcdef"), {}, {})[-1]]
    ['pair', 'pair', 'pair']
    """
    rounds = [[Run("intra", (node,)) for node in nodes], [Run("loopback", (node,), loopback_env) for node in nodes]]
    rounds += [[Run("pair", pair) for pair in pairs] for pairs in pair_rounds(nodes)]
    leaves: dict[str, list[str]] = {}
    for node in nodes:
        if node in switch_of:
            leaves.setdefault(switch_of[node], []).append(node)
    if groups := [Run("group", tuple(members)) for members in leaves.values() if len(members) > 2]:
        rounds.append(groups)
    return rounds


def main() -> int:
    results = cfg.paths
    # srun passes these to every rank; run_round names each rank's debug log, and finalize scans them for warnings
    results.nccl_logs.mkdir(parents=True, exist_ok=True)
    os.environ["NCCL_DEBUG"] = cfg.nccl.debug
    env = read_json(cfg.paths.env) or {}
    job_id = os.environ["SLURM_JOB_ID"]
    nodes = list(Hostlist.expand(os.environ["SLURM_JOB_NODELIST"]))
    # One rank per GPU
    launch = Launch(env.get("mpi", "pmix"), Path(env["nccl_tests"]) / cfg.nccl.binary, cfg.nccl.args, cfg.gpu.count)
    start = time.monotonic()
    finished: list[dict[str, Any]] = []
    out: dict[str, Json] = {"job_id": job_id, "nodes": nodes, "mpi": launch.mpi, "tests": finished}
    for tests in plan(nodes, cfg.nccl.loopback_env, node_switches()):
        timeout = cfg.nccl.timeout_s[tests[0].kind]
        finished += run_round(tests, launch, results.group_logs(job_id), timeout, results.nccl_logs)
        out["elapsed_s"] = round(time.monotonic() - start, 1)  # the phase's wall time; a round's tests overlap
        write_json(results.group(job_id), out)  # after each round, in case the job is killed
        print(
            f"nccl {tests[0].kind}: "
            + ", ".join(
                f"{result['nodes'][0]}={result['peak_busbw']}({result['status']})" for result in finished[-len(tests) :]
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
