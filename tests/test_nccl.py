import sys
import time
from pathlib import Path

from healthcheck import nccl
from healthcheck.check import Availability
from healthcheck.config import Nccl, NcclTest, Paths
from healthcheck.util.cmd import Cmd

ALL_REDUCE = r"""
# nThread 1 nGpus 1 minBytes 134217728 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 123456 on    rg31601 device  0 [0x19] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 123457 on    rg31601 device  1 [0x3b] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
   134217728      33554432     float     sum      -1    731.2  183.56  275.34      0    729.9  183.88  275.83      0
   268435456      67108864     float     sum      -1   1398.4  191.96  287.95      0   1397.1  192.14  288.21      0
   536870912     134217728     float     sum      -1   2741.0  195.87  293.80      0   2739.6  195.97  293.95      0
  1073741824     268435456     float     sum      -1   5423.7  197.97  296.96      0   5420.2  198.10  297.15      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 288.65
#
"""


def test_parse_all_reduce():
    parsed = nccl.parse(ALL_REDUCE)
    assert len(parsed["rows"]) == 4
    assert parsed["peak_busbw"] == 296.96
    assert parsed["avg_busbw"] == 288.65
    assert parsed["wrong"] == 0


def test_parse_failed_run():
    assert nccl.parse("srun: error: rg31601: task 0: Exited with exit code 1\n")["peak_busbw"] is None


def test_pair_rounds_cover_ring_once_per_round():
    for count in range(2, 12):
        nodes = [f"n{index}" for index in range(count)]
        rounds = nccl.pair_rounds(nodes)
        pairs = [pair for round_ in rounds for pair in round_]
        assert len(rounds) <= 3
        for round_ in rounds:
            members = [node for pair in round_ for node in pair]
            assert len(members) == len(set(members))
        if count > 2:
            assert len(pairs) == count  # every node in exactly two pairs
            assert all(sum(node in pair for pair in pairs) == 2 for node in nodes)


def test_plan_single_node_has_no_pairs_or_group():
    kinds = [test.kind for tests in nccl.plan(["a"]) for test in tests]
    assert kinds == ["intra", "loopback"]
    rounds = nccl.plan(["a", "b", "c"])
    assert {test.kind for tests in rounds for test in tests} == {"intra", "loopback", "pair", "group"}


def test_plan_collectives_span_the_whole_job():
    # --job-per picks the job's nodes: a leaf switch's or the cluster's
    nodes = [f"n{index}" for index in range(21)]
    rounds = nccl.plan(nodes)
    assert [test.nodes for test in rounds[-1]] == [tuple(nodes)]  # group


def test_plan_skips_disabled_tests():
    off = NcclTest(enabled=False)
    settings = Nccl()
    alltoall = settings.alltoall.model_copy(update={"enabled": True})
    only_alltoall = settings.model_copy(
        update={"intra": off, "loopback": off, "pair": off, "group": off, "alltoall": alltoall}
    )
    rounds = nccl.plan(["a", "b"], only_alltoall)
    assert [[test.label for test in tests] for tests in rounds] == [["alltoall-a-b"]]
    assert nccl.plan(["only-one"], only_alltoall) == []
    # Alongside the other tests, two nodes do not run alltoall
    alongside = settings.model_copy(update={"alltoall": alltoall})
    assert "alltoall" not in [test.kind for tests in nccl.plan(["a", "b"], alongside) for test in tests]


def test_each_test_uses_its_binary_args_and_env():
    launch = nccl.Launch("pmix", Path("/opt/build"), 4)
    settings = Nccl()
    rounds = nccl.plan(["a", "b", "c"])
    by_kind = {test.kind: test for tests in rounds for test in tests}
    assert launch.command(by_kind["group"])[-6:] == [
        "/opt/build/all_reduce_perf",
        "--minbytes=128M",
        "--maxbytes=1G",
        "--stepfactor=2",
        "--warmup_iters=5",
        "--iters=20",
    ]
    alltoall = nccl.Run(settings.alltoall.name, ("a", "b", "c"), settings.alltoall)
    assert launch.command(alltoall)[-5:] == [
        "/opt/build/alltoall_perf",
        "--minbytes=1G",
        "--maxbytes=1G",
        "--warmup_iters=2",
        "--iters=5",
    ]
    assert by_kind["loopback"].env == {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
    }
    assert by_kind["intra"].env == {}


def test_run_round_terminates_stragglers(tmp_path):
    class Sleeper:
        """Stands in for Launch: a test that never finishes."""

        def command(self, test):
            return ["sleep", "30"]

    logs = tmp_path / "nccl"
    logs.mkdir()
    (logs / "pair-a-b.b.7.log").touch()  # a remote rank that may outlive srun: its log is kept
    start = time.monotonic()
    [result] = nccl.run_round([nccl.Run("pair", ("a", "b"))], Sleeper(), tmp_path, timeout=0.5, debug_dir=logs)
    assert result["status"] == "timeout"
    assert time.monotonic() - start < 10
    assert (logs / "pair-a-b.b.7.log").exists()


def test_run_round_names_debug_logs_and_removes_empty_ones(tmp_path):
    """Two ranks fill in NCCL_DEBUG_FILE's %h and %p as NCCL does; only rank 0 writes, its version line."""
    rank = tmp_path / "rank.py"
    rank.write_text(
        "import os, sys\n"
        'path = os.environ["NCCL_DEBUG_FILE"].replace("%h", sys.argv[1]).replace("%p", sys.argv[2])\n'
        'open(path, "w").write(" ".join(sys.argv[3:]))\n'
    )

    class Ranks:
        """Stands in for Launch."""

        def command(self, test):
            host = test.nodes[0]
            return [
                "sh",
                "-c",
                f"{sys.executable} {rank} {host} 100 NCCL version && {sys.executable} {rank} {host} 101",
            ]

    logs = tmp_path / "nccl"
    logs.mkdir()
    (logs / "intra-b.b.7.log").touch()  # another job's rank, still running
    nccl.run_round([nccl.Run("intra", ("a",))], Ranks(), tmp_path / "groups", timeout=30, debug_dir=logs)
    assert sorted(path.name for path in logs.iterdir()) == ["intra-a.a.100.log", "intra-b.b.7.log"]
    assert (logs / "intra-a.a.100.log").read_text() == "NCCL version"


def test_mpi_plugin(monkeypatch):
    def listed(stdout="", stderr="", returncode=0):
        monkeypatch.setattr(nccl, "run", lambda cmd, **_: Cmd(returncode, stdout, stderr))

    listed("MPI plugin types are...\n\tpmix\n\tpmi2\n\tnone\n")
    assert nccl.mpi_plugin() == "pmix"
    listed("\tnone\n\tpmi2\n")
    assert nccl.mpi_plugin() == "pmi2"
    listed("srun: MPI types are...\nsrun: none\n")
    assert nccl.mpi_plugin() == "none"
    listed("", "srun: pmix\nsrun: pmi2\n")
    assert nccl.mpi_plugin() == "pmix"  # some srun versions print the list on stderr
    listed("", "srun: error: MPI plugin pmix not available\n", 1)
    assert nccl.mpi_plugin() == "none"


def _off() -> Nccl:
    disabled = NcclTest(enabled=False)
    return Nccl(intra=disabled, loopback=disabled, pair=disabled, group=disabled, alltoall=disabled)


def test_available_without_build(monkeypatch, tmp_path):
    monkeypatch.delenv("HC_NCCL_TESTS", raising=False)
    found = nccl.HealthCheck(Nccl(), Paths(root=tmp_path), 4).available()
    assert found == Availability(False, "HC_NCCL_TESTS is not set")


def test_available_missing_binary(monkeypatch, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setenv("HC_NCCL_TESTS", str(tmp_path))

    def probed():
        raise AssertionError("probed")

    monkeypatch.setattr(nccl, "mpi_plugin", probed)
    found = nccl.HealthCheck(Nccl(), Paths(root=tmp_path), 4).available()
    assert found.available is False
    assert found.reason == f"missing all_reduce_perf in {build}"


def test_available_without_mpi(monkeypatch, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "all_reduce_perf").write_bytes(b"")
    monkeypatch.setenv("HC_NCCL_TESTS", str(tmp_path))
    monkeypatch.setattr(nccl, "mpi_plugin", lambda: "none")
    found = nccl.HealthCheck(Nccl(), Paths(root=tmp_path), 4).available()
    assert found == Availability(False, "no MPI plugin")


def test_write_unavailable_keeps_mpi_none(monkeypatch, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "all_reduce_perf").write_bytes(b"")
    monkeypatch.setenv("HC_NCCL_TESTS", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "cn-l001")
    monkeypatch.setattr(nccl, "mpi_plugin", lambda: "none")
    check = nccl.HealthCheck(Nccl(), Paths(root=tmp_path / "run"), 4)
    written = check.write_unavailable(check.available())
    assert written["mpi"] == "none"
    assert "status" not in written
    assert (tmp_path / "run" / "groups" / "9.json").is_file()


def test_start_skips_when_disabled(monkeypatch, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "all_reduce_perf").write_bytes(b"")
    (build / "alltoall_perf").write_bytes(b"")
    monkeypatch.setenv("HC_NCCL_TESTS", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "cn-l001")

    def probed():
        raise AssertionError("probed")

    monkeypatch.setattr(nccl, "mpi_plugin", probed)
    written = nccl.start(nccl.HealthCheck(_off(), Paths(root=tmp_path / "run"), 4))
    assert written["tests"] == []
    assert "status" not in written
    assert "mpi" not in written
    assert written["elapsed_s"] == 0
