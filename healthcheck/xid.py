"""NVML Xid events for the whole job, standing in for the root-only kernel log's NVRM lines.

The watcher appends one JSON object per line to <run>/xid/<host>.jsonl. Phase C reads it back.
"""

import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pynvml

from healthcheck.config import cfg
from healthcheck.util.file import JsonDict
from healthcheck.util.nvml import session
from healthcheck.util.status import Status


def watch() -> None:
    """Record events until the stop file appears.

    A stop file rather than a signal: srun answers SIGTERM by killing the step, before "stopped" is written.
    """
    if not cfg.logs.xid_watch:
        return
    path = cfg.paths.xid(cfg.host.name)
    stop_file = cfg.paths.xid_stop(os.environ.get("SLURM_JOB_ID", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def on_term(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    with open(path, "a", buffering=1) as xid_log:

        def emit(**event: Any) -> None:
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            xid_log.write(json.dumps({"time": stamp, **event}) + "\n")

        with session():
            events = pynvml.nvmlEventSetCreate()
            for index in range(pynvml.nvmlDeviceGetCount()):
                try:
                    pynvml.nvmlDeviceRegisterEvents(
                        pynvml.nvmlDeviceGetHandleByIndex(index), pynvml.nvmlEventTypeXidCriticalError, events
                    )
                except pynvml.NVMLError as error:
                    emit(event="error", gpu=index, error=str(error))
            emit(event="started")
            while not stop and not stop_file.exists():
                try:
                    data = pynvml.nvmlEventSetWait_v2(events, 1000)
                except pynvml.NVMLError_Timeout:
                    continue
                except pynvml.NVMLError as error:
                    emit(event="error", error=str(error))
                    time.sleep(1)
                    continue
                emit(event="xid", gpu=pynvml.nvmlDeviceGetIndex(data.device), xid=int(data.eventData))
            emit(event="stopped")
            pynvml.nvmlEventSetFree(events)


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


if __name__ == "__main__":
    watch()
