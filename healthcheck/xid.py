"""NVML Xid events for the whole job, standing in for the root-only kernel log's NVRM lines.

watch() appends one JSON object per line to <run>/xid/<host>.jsonl. Phase C reads it back with events().
"""

import json
import signal
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pynvml

from healthcheck.config import cfg
from healthcheck.util.file import JsonDict
from healthcheck.util.nvml import event_set, session
from healthcheck.util.status import Status

type Event = Literal["started", "stopped", "xid", "error"]


class EmitFn(Protocol):
    """
    Appends one event to the watcher's log. An xid has gpu and xid; an error has error, and gpu when one GPU's
    registration failed.
    """

    def __call__(
        self,
        *,
        event: Event,
        gpu: int | None = None,
        xid: int | None = None,
        error: str | None = None
    ) -> None: ...


def watch() -> None:
    """Record Xid events to <run>/xid/<host>.jsonl until SIGUSR1, which the batch step sends after NCCL.

    Runs as its own srun step on every node, alongside the others. With logs.xid_watch off, it returns at once.
    """
    if not cfg.logs.xid_watch:
        return
    stop = threading.Event()
    signal.signal(signal.SIGUSR1, lambda _signum, _frame: stop.set())
    signal.siginterrupt(signal.SIGUSR1, False)  # NVML's 1 s wait finishes instead of failing with EINTR
    path = cfg.paths.xid(cfg.host.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as log:

        def emit(*, event: Event, gpu: int | None = None, xid: int | None = None, error: str | None = None) -> None:
            fields = {"gpu": gpu, "xid": xid, "error": error}
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            present = {key: value for key, value in fields.items() if value is not None}
            log.write(json.dumps({"time": stamp, "event": event, **present}) + "\n")

        try:
            with session(), event_set() as events:
                _register(events, emit)
                emit(event="started")
                while not stop.is_set():
                    _poll(events, emit, stop)
                emit(event="stopped")
        except Exception as error:  # noqa: BLE001 - the log keeps it; the other steps keep running
            emit(event="error", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()


def _register(events: Any, emit: EmitFn) -> None:
    for index in range(pynvml.nvmlDeviceGetCount()):
        try:
            pynvml.nvmlDeviceRegisterEvents(
                pynvml.nvmlDeviceGetHandleByIndex(index), pynvml.nvmlEventTypeXidCriticalError, events
            )
        except pynvml.NVMLError as error:
            emit(event="error", gpu=index, error=str(error))


def _poll(events: Any, emit: EmitFn, stop: threading.Event) -> None:
    """One wait. A timeout returns so the caller can check stop; any other NVML error is logged."""
    try:
        data = pynvml.nvmlEventSetWait_v2(events, 1000)
    except pynvml.NVMLError_Timeout:
        return
    except pynvml.NVMLError as error:
        emit(event="error", error=str(error))
        stop.wait(1)
        return
    emit(event="xid", gpu=pynvml.nvmlDeviceGetIndex(data.device), xid=int(data.eventData))


def events(path: Path, ignore: set[int]) -> JsonDict:
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
