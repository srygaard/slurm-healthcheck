"""NVML Xid events for the whole job, standing in for the root-only kernel log's NVRM lines.

A watcher thread appends one JSON object per line to <run>/xid/<host>.jsonl. Phase C reads it back after stop().
"""

import json
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import pynvml

from healthcheck.config import cfg
from healthcheck.util.file import JsonDict
from healthcheck.util.nvml import session
from healthcheck.util.status import Status


@contextmanager
def _event_set() -> Iterator[Any]:
    """An NVML event set. Enter it inside session(), so free runs before nvmlShutdown."""
    events = pynvml.nvmlEventSetCreate()
    try:
        yield events
    finally:
        pynvml.nvmlEventSetFree(events)


class Watcher:
    """A thread that records Xid events until stop().

    start() returns at once. With logs.xid_watch off, both calls do nothing.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log: TextIO | None = None

    def start(self) -> None:
        if not cfg.logs.xid_watch or self._thread is not None:
            return
        path = cfg.paths.xid(cfg.host.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(path, "a", buffering=1)  # noqa: SIM115 -- open until stop(), across the watcher thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, name="xid", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to write "stopped" and wait for it. The NVML wait is 1s."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=5)
        if self._log is not None:
            self._log.close()
            self._log = None

    def _emit(self, **event: Any) -> None:
        """Append one event. start() has opened the log; stop() closes it after the thread returns."""
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        if self._log is not None:
            self._log.write(json.dumps({"time": stamp, **event}) + "\n")

    def _register(self, events: Any) -> None:
        for index in range(pynvml.nvmlDeviceGetCount()):
            try:
                pynvml.nvmlDeviceRegisterEvents(
                    pynvml.nvmlDeviceGetHandleByIndex(index), pynvml.nvmlEventTypeXidCriticalError, events
                )
            except pynvml.NVMLError as error:
                self._emit(event="error", gpu=index, error=str(error))

    def _poll(self, events: Any) -> None:
        """One wait. A timeout returns so the caller can check stop; any other NVML error is logged."""
        try:
            data = pynvml.nvmlEventSetWait_v2(events, 1000)
        except pynvml.NVMLError_Timeout:
            return
        except pynvml.NVMLError as error:
            self._emit(event="error", error=str(error))
            self._stop.wait(1)
            return
        self._emit(event="xid", gpu=pynvml.nvmlDeviceGetIndex(data.device), xid=int(data.eventData))

    def _watch(self) -> None:
        try:
            with session(), _event_set() as events:
                self._register(events)
                self._emit(event="started")
                while not self._stop.is_set():
                    self._poll(events)
                self._emit(event="stopped")
        except Exception as error:  # noqa: BLE001 - the log keeps it; the node steps keep running
            self._emit(event="error", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()


def events(path: Path, ignore: list[int]) -> JsonDict:
    """The watcher's log: Xids, apart from ignore, and any errors it recorded."""
    try:
        recorded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return {"status": Status.UNAVAILABLE, "error": "watcher wrote nothing"}
    xids = [event for event in recorded if event.get("event") == "xid"]
    return {
        "status": Status.OK if any(event.get("event") == "started" for event in recorded) else Status.UNAVAILABLE,
        "events": [xid for xid in xids if xid["xid"] not in ignore],
        "ignored": [xid for xid in xids if xid["xid"] in ignore],
        "errors": [event for event in recorded if event.get("event") == "error"],
    }
