import time

from healthcheck import nccl

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
    kinds = [test.kind for tests in nccl.plan(["a"], {}, {}) for test in tests]
    assert kinds == ["intra", "loopback"]
    kinds = {test.kind for tests in nccl.plan(["a", "b", "c"], {}, {}) for test in tests}
    assert kinds == {"intra", "loopback", "pair", "group"}


def test_run_round_terminates_stragglers(tmp_path):
    class Sleeper:
        """Stands in for Launch: a test that never finishes."""

        def command(self, test):
            return ["sleep", "30"]

    start = time.monotonic()
    [result] = nccl.run_round([nccl.Run("pair", ("a", "b"))], Sleeper(), tmp_path, timeout=0.5)
    assert result["status"] == "timeout"
    assert time.monotonic() - start < 10
