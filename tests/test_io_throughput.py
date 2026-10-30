from healthcheck import share
from healthcheck.util.io_throughput import read_throughput

BLOCK = 16 * 2**20


def test_scratch_read_loops_for_the_whole_window(tmp_path):
    path = tmp_path / "f0"
    path.write_bytes(b"\0" * (2 * BLOCK))
    result = read_throughput(path, float("inf"), max_s=0.3, block=BLOCK, loop=True)
    assert result["bytes"] > path.stat().st_size  # wrapped around at least once
    assert result["seconds"] >= 0.3


def test_read_without_loop_stops_at_eof(tmp_path):
    path = tmp_path / "f0"
    path.write_bytes(b"\0" * BLOCK)
    assert read_throughput(path, float("inf"), max_s=5, block=BLOCK)["bytes"] == BLOCK


def test_empty_file_does_not_spin(tmp_path):
    path = tmp_path / "f0"
    path.touch()
    assert read_throughput(path, float("inf"), max_s=5, block=BLOCK, loop=True)["bytes"] == 0


def test_write_random_skips_complete_files(tmp_path):
    path = share.file_path(tmp_path, 0)
    assert share.write_random(path, BLOCK + 1, BLOCK)
    assert path.stat().st_size == 2 * BLOCK  # whole blocks
    assert not share.write_random(path, BLOCK, BLOCK)
    assert not path.with_name("f0.tmp").exists()
