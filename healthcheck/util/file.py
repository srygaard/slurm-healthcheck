"""JSON files that several processes write and read, and single-value system files."""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# What a step returns and write_json takes. Mapping and Sequence, unlike dict and list, accept a dict[str, float].
# Values read back stay Any: indexing into this union would need a narrowing check at every level.
type Json = Mapping[str, Json] | Sequence[Json] | str | int | float | bool | None
type JsonDict = Mapping[str, Json]


def write_json(path: Path, data: Json) -> None:
    """Write atomically, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        json.dump(data, tmp, indent=1, default=str)
    os.replace(tmp.name, path)


def read_json(path: Path) -> Any:
    """Parsed JSON, or None for a missing or empty (never-written) file."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    return json.loads(text) if text.strip() else None


def read_sys(path: str | Path) -> str | None:
    """A sysfs/procfs value, stripped; None if it's missing or unreadable."""
    try:
        with open(path) as sys_file:
            return sys_file.read().strip()
    except (OSError, ValueError):
        return None
