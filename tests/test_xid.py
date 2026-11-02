import json

from healthcheck import xid
from healthcheck.util.status import Status


def test_events_drops_ignored_xids_and_keeps_errors(tmp_path):
    path = tmp_path / "cn-l001.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"event": "started", "time": "t0"},
                {"event": "xid", "gpu": 1, "xid": 43, "time": "t1"},
                {"event": "xid", "gpu": 0, "xid": 79, "time": "t2"},
                {"event": "error", "gpu": 2, "error": "register failed"},
                {"event": "stopped", "time": "t3"},
            ]
        )
    )
    result = xid.events(path, ignore=[43])
    assert result["status"] == Status.OK
    assert result["events"] == [{"event": "xid", "gpu": 0, "xid": 79, "time": "t2"}]
    assert result["ignored"] == [{"event": "xid", "gpu": 1, "xid": 43, "time": "t1"}]
    assert result["errors"] == [{"event": "error", "gpu": 2, "error": "register failed"}]


def test_events_when_the_watcher_wrote_nothing(tmp_path):
    assert xid.events(tmp_path / "missing.jsonl", ignore=[])["status"] == Status.UNAVAILABLE
