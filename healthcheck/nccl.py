"""Step 3c: nccl-tests across the job's nodes, launched as concurrent srun steps from the batch step.

Rounds, each a set of concurrent steps on disjoint nodes:
  intra     every node alone: NVLink/NVSwitch
  loopback  every node alone with P2P/SHM off, so traffic goes through its NICs
  pairs     ring neighbours (i, i+1): a bad NIC shows up in both of a node's pairs
  group     all the job's nodes together: a leaf switch's with --job-per switch, every node with --job-per cluster
  alltoall  all the job's nodes with alltoall_perf, every rank sending to every other
"""

import os
import re
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from healthcheck.check import Availability, Finding, HealthCheckBase, Report, Severity, median
from healthcheck.config import Nccl, NcclTest, Paths, cfg
from healthcheck.util.cmd import run
from healthcheck.util.file import Json, JsonDict, write_json
from healthcheck.util.hostlist import ByHost, Hostlist
from healthcheck.util.status import Status, ok


class Run(NamedTuple):
    kind: str
    nodes: tuple[str, ...]
    test: NcclTest = NcclTest()

    @property
    def label(self) -> str:
        return f"{self.kind}-{self.nodes[0]}" + (f"-{self.nodes[-1]}" if len(self.nodes) > 1 else "")

    @property
    def env(self) -> dict[str, str]:
        return self.test.env


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
    """How every test of a run is started: one nccl-tests build directory and the ranks per node.

    The binary and arguments come from the test.
    """

    mpi: str
    build: Path
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
            str(self.build / test.test.binary),
            *test.test.args,
        ]


def run_round(
    tests: list[Run],
    launch: Launch,
    out_dir: Path,
    timeout: float,
    debug_dir: Path,
) -> list[dict[str, Any]]:
    """
    Run tests concurrently; a straggler past timeout is terminated and reported as such.

    Each rank writes NCCL's debug log to <debug_dir>/<label>.<host>.<pid>.log, and the empty ones of tests that
    finished are removed after it: at NCCL_DEBUG=WARN a healthy rank writes nothing, rank 0 only the version. A
    timed-out test's are all kept, since its remote ranks may outlive srun and still be writing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    start = time.monotonic()
    for test in tests:
        log = out_dir / f"{test.label}.txt"
        env = os.environ | test.env
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
        if len(ended) == len(procs) or now - start >= timeout:
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
                "rows": [
                    {
                        "size": row["size"],
                        "busbw": row["busbw"]
                    }
                    for row in parsed["rows"]
                ],
                "wrong": parsed["wrong"],
                "log": str(log),
            }
        )
    # Only this round's finished tests: the run's other jobs share debug_dir, and their ranks may still be writing
    for index, (test, _, _) in enumerate(procs):
        for path in debug_dir.glob(f"{test.label}.*.log") if index in ended else ():
            if path.stat().st_size == 0:
                path.unlink(missing_ok=True)
    return results


def plan(nodes: list[str], settings: Nccl | None = None) -> list[list[Run]]:
    """
    Rounds of concurrent tests, skipping kinds whose NcclTest is disabled. The per-node and pair rounds take about
    the same time at any node count; the group and alltoall tests span the whole job, when it has more than two
    nodes, so they slow down as it grows. Alltoall alone runs from two nodes; alongside the other tests it waits
    for more than two, the same rule as group.

    >>> [[test.label for test in tests] for tests in plan(["a", "b", "c", "d"])][2:]
    [['pair-a-b', 'pair-c-d'], ['pair-b-c', 'pair-d-a'], ['group-a-d']]
    >>> [test.kind for tests in plan(["a", "b"]) for test in tests]
    ['intra', 'intra', 'loopback', 'loopback', 'pair']
    >>> plan([])
    []
    """
    chosen = cfg.nccl if settings is None else settings
    rounds: list[list[Run]] = []
    if chosen.intra.enabled and nodes:
        rounds.append([Run(chosen.intra.name, (node,), chosen.intra) for node in nodes])
    if chosen.loopback.enabled and nodes:
        rounds.append([Run(chosen.loopback.name, (node,), chosen.loopback) for node in nodes])
    if chosen.pair.enabled and len(nodes) >= 2:
        rounds += [[Run(chosen.pair.name, pair, chosen.pair) for pair in pairs] for pairs in pair_rounds(nodes)]
    whole = tuple(nodes)
    if chosen.group.enabled and len(nodes) > 2:
        rounds.append([Run(chosen.group.name, whole, chosen.group)])
    # Two nodes already have a pair. Alltoall joins them only when it is the only test configured.
    only_alltoall = not (
        chosen.intra.enabled or chosen.loopback.enabled or chosen.pair.enabled or chosen.group.enabled
    )
    if chosen.alltoall.enabled and len(nodes) >= (2 if only_alltoall else 3):
        rounds.append([Run(chosen.alltoall.name, whole, chosen.alltoall)])
    return rounds


def mpi_plugin() -> str:
    """The srun --mpi plugin to use: pmix if listed, else pmi2, else none.

    No plugin skips NCCL. An attempt would default to pmix and fail, while the other phases still run.
    """
    out = run(["srun", "--mpi=list"], timeout=30)
    if out.returncode:
        return "none"
    listed = set(re.findall(r"\b(pmix|pmi2)\b", f"{out.stdout}\n{out.stderr}"))
    for name in ("pmix", "pmi2"):
        if name in listed:
            return name
    return "none"


class HealthCheck(HealthCheckBase):
    def __init__(self, settings: Nccl, paths: Paths, ranks: int) -> None:
        self.settings = settings
        self.paths = paths
        self.ranks = ranks
        self._mpi: str | None = None
        self._build: Path | None = None

    def _tests(self) -> tuple[NcclTest, ...]:
        settings = self.settings
        return (settings.intra, settings.loopback, settings.pair, settings.group, settings.alltoall)

    @property
    def enabled(self) -> bool:
        return any(test.enabled for test in self._tests())

    def available(self) -> Availability:
        """The nccl-tests build for every enabled kind, and an srun MPI plugin."""
        root = os.environ.get("HC_NCCL_TESTS")
        if not root:
            return Availability(False, "HC_NCCL_TESTS is not set")
        build = Path(root) / "build"
        self._build = build
        missing = list(
            dict.fromkeys(
                test.binary for test in self._tests() if test.enabled and not (build / test.binary).is_file()
            )
        )
        if missing:
            return Availability(False, f"missing {', '.join(missing)} in {build}")
        self._mpi = mpi_plugin()
        if self._mpi == "none":
            return Availability(False, "no MPI plugin")
        return Availability(True)

    def _job(self) -> tuple[str, list[str]]:
        return os.environ["SLURM_JOB_ID"], list(Hostlist.expand(os.environ["SLURM_JOB_NODELIST"]))

    def _save(self, out: dict[str, Json]) -> None:
        self.paths.nccl_logs.mkdir(parents=True, exist_ok=True)
        write_json(self.paths.group(str(out["job_id"])), out)

    def write_skipped(self) -> JsonDict:
        """The group file when every kind is disabled. No plugin probe and no launch."""
        job_id, nodes = self._job()
        out: dict[str, Json] = {"job_id": job_id, "nodes": nodes, "kinds": [], "tests": [], "elapsed_s": 0}
        self._save(out)
        print(f"nccl: nothing to run ({len(nodes)} node(s))")
        return out

    def write_unavailable(self, found: Availability) -> JsonDict:
        """The group file when the tool is missing. A missing MPI plugin keeps the report's mpi=none warning."""
        job_id, nodes = self._job()
        out: dict[str, Json] = {"job_id": job_id, "nodes": nodes, "kinds": [], "tests": [], "elapsed_s": 0}
        if self._mpi == "none":
            out["mpi"] = "none"
            message = "skipped, no MPI plugin"
        else:
            out["status"] = Status.UNAVAILABLE
            out["error"] = found.reason
            message = f"unavailable, {found.reason}"
        self._save(out)
        print(f"nccl: {message} ({len(nodes)} node(s))")
        return out

    def collect(self) -> JsonDict:
        results = self.paths
        # srun passes these to every rank; run_round names each rank's debug log, and finalize scans them for warnings
        results.nccl_logs.mkdir(parents=True, exist_ok=True)
        os.environ["NCCL_DEBUG"] = self.settings.debug
        job_id, nodes = self._job()
        # No plugin means no rounds: one empty result either way, and the report warns when mpi is none.
        mpi = self._mpi if self._mpi is not None else mpi_plugin()
        rounds = [] if mpi == "none" else plan(nodes, self.settings)
        finished: list[dict[str, Any]] = []
        out: dict[str, Json] = {
            "job_id": job_id,
            "nodes": nodes,
            "mpi": mpi,
            "kinds": list(dict.fromkeys(tests[0].kind for tests in rounds if tests)),
            "tests": finished,
        }
        if not rounds:
            out["elapsed_s"] = 0
            write_json(results.group(job_id), out)
            print(f"nccl: {'skipped, no MPI plugin' if mpi == 'none' else 'nothing to run'} ({len(nodes)} node(s))")
            return out
        # One rank per GPU. Each test names its own binary and arguments.
        build = self._build if self._build is not None else Path(os.environ["HC_NCCL_TESTS"]) / "build"
        launch = Launch(mpi, build, self.ranks)
        started = time.monotonic()
        for tests in rounds:
            finished += run_round(tests, launch, results.group_logs(job_id), tests[0].test.timeout_s, results.nccl_logs)
            out["elapsed_s"] = round(time.monotonic() - started, 1)  # the phase's wall time; a round's tests overlap
            write_json(results.group(job_id), out)  # after each round, in case the job is killed
            print(
                f"nccl {tests[0].kind}: "
                + ", ".join(
                    f"{result['nodes'][0]}={result['peak_busbw']}({result['status']})"
                    for result in finished[-len(tests) :]
                ),
            )
        return out

    @staticmethod
    def report(groups: list[dict[str, Any]], cfg: Nccl, name: str) -> Report:
        out: ByHost[list[Finding]] = defaultdict(list)
        for group in groups:
            if group.get("mpi") == "none":
                for node in group.get("nodes", []):
                    out[node].append(Finding("warn", name, "skipped: no MPI plugin"))
            elif group.get("status") == Status.UNAVAILABLE:
                detail = f"unavailable: {group.get('error', '')}".strip(": ")
                for node in group.get("nodes", []):
                    out[node].append(Finding("warn", name, detail))
        tests = [
            test
            for group in groups
            for test in group["tests"]
        ]
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for test in tests:
            by_kind[test["kind"]].append(test)
        kind_medians = {
            kind: median(
                test["peak_busbw"]
                for test in kind_tests
                if ok(test)
            )
            for kind, kind_tests in by_kind.items()
        }
        for spec in (cfg.intra, cfg.loopback):
            for test in by_kind[spec.name]:
                if not ok(test):
                    detail = f"{test['status']} ({test['log']})"
                    out[test["nodes"][0]].append(Finding("fail", f"{name}.{spec.name}", detail))
                elif test["peak_busbw"] < spec.min_busbw_GBps:
                    detail = f"busbw {test['peak_busbw']} < {spec.min_busbw_GBps}"
                    out[test["nodes"][0]].append(Finding("fail", f"{name}.{spec.name}", detail))

        # A bad node makes both of its ring pairs slow; its healthy neighbours see only one. A node whose every pair is
        # bad fails too: with two nodes there is one pair, and nothing tells the two apart.
        pair_floor = _floor(cfg.pair, kind_medians)
        slow: ByHost[list[str]] = defaultdict(list)
        pairs_of: Counter[str] = Counter()
        for test in by_kind[cfg.pair.name]:
            pairs_of.update(test["nodes"])
            if not ok(test) or test["peak_busbw"] < pair_floor:
                for node in test["nodes"]:
                    other = next(peer for peer in test["nodes"] if peer != node)
                    slow[node].append(f"{other}: {test['status']} {test['peak_busbw']}")
        for node, partners in slow.items():
            severity: Severity = "fail" if len(partners) >= 2 or len(partners) == pairs_of[node] else "warn"
            detail = f"{len(partners)}/{pairs_of[node]} slow pairs (< {pair_floor:.1f}): {'; '.join(partners)}"
            out[node].append(Finding(severity, f"{name}.{cfg.pair.name}", detail))

        for spec in (cfg.group, cfg.alltoall):
            floor = _floor(spec, kind_medians)
            op = spec.binary.removesuffix("_perf")
            for test in by_kind[spec.name]:
                if not ok(test) or test["peak_busbw"] < floor:
                    # A hung group test with no node already failing for it can't be blamed on one, so every node fails.
                    # Alltoall's timeout_severity is warn: a full-cluster timeout names no node.
                    blamed = any(
                        finding.severity == "fail"
                        for node in test["nodes"]
                        for finding in out.get(node, [])
                    )
                    severity = spec.timeout_severity if test["status"] == Status.TIMEOUT and not blamed else "warn"
                    detail = (
                        f"{len(test['nodes'])}-node {op} {test['status']} "
                        f"busbw {test['peak_busbw']} (floor {floor:.1f})"
                    )
                    for node in test["nodes"]:
                        out[node].append(Finding(severity, f"{name}.{spec.name}", detail))
        prefixed = {f"nccl_{kind}_busbw_GBps": value for kind, value in kind_medians.items()}
        return Report(dict(out), prefixed)


def _floor(spec: NcclTest, kind_medians: dict[str, float | None]) -> float:
    return max(spec.min_busbw_GBps, (kind_medians.get(spec.name) or 0) * (spec.min_frac_of_median or 0))


def start(check: HealthCheck) -> JsonDict:
    """Skip a disabled check, record a missing tool, or collect. Writes the group file either way."""
    if not check.enabled:
        return check.write_skipped()
    found = check.available()
    if not found.available:
        return check.write_unavailable(found)
    return check.collect()


def main() -> int:
    start(HealthCheck(cfg.nccl, cfg.paths, cfg.gpu.count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
