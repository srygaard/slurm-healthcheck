import contextlib
import json
import os
import signal

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
    result = xid.events(path, ignore={43})
    assert result["status"] == Status.OK
    assert result["events"] == [{"event": "xid", "gpu": 0, "xid": 79, "time": "t2"}]
    assert result["ignored"] == [{"event": "xid", "gpu": 1, "xid": 43, "time": "t1"}]
    assert result["errors"] == [{"event": "error", "gpu": 2, "error": "register failed"}]


def test_events_when_the_watcher_wrote_nothing(tmp_path):
    assert xid.events(tmp_path / "missing.jsonl", ignore=set())["status"] == Status.UNAVAILABLE


def test_watch_logs_until_sigusr1(monkeypatch, tmp_path):
    """The batch step's SIGUSR1 ends the loop and the log closes with "stopped", with no GPU."""

    class Paths:
        def xid(self, host):
            return tmp_path / f"{host}.jsonl"

    class Logs:
        xid_watch = True

    class Host:
        name = "n1"

    class Cfg:
        paths = Paths()
        logs = Logs()
        host = Host()

    polls = []

    def poll(events, emit, stop):
        polls.append(events)
        if len(polls) == 2:
            os.kill(os.getpid(), signal.SIGUSR1)

    monkeypatch.setattr(xid, "cfg", Cfg())
    monkeypatch.setattr(xid, "session", contextlib.nullcontext)
    monkeypatch.setattr(xid, "event_set", contextlib.nullcontext)
    monkeypatch.setattr(xid, "_register", lambda events, emit: None)
    monkeypatch.setattr(xid, "_poll", poll)
    handler = signal.getsignal(signal.SIGUSR1)
    try:
        xid.watch()
    finally:
        signal.signal(signal.SIGUSR1, handler)
    assert len(polls) == 2
    events = [json.loads(line)["event"] for line in (tmp_path / "n1.jsonl").read_text().splitlines()]
    assert events == ["started", "stopped"]
