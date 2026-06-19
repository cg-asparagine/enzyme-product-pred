"""Filesystem helpers: run-directory naming and robust JSON (de)serialization."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def new_run_id(model_name: str, git_commit: str = "", timestamp: datetime | None = None) -> str:
    """A sortable, unique-ish run id: ``<UTC-timestamp>__<model>__<git7>``."""
    ts = (timestamp or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    short = (git_commit or "nogit")[:7]
    return f"{ts}__{_slugify(model_name)}__{short}"


def new_run_dir(
    root: str | Path,
    model_name: str,
    git_commit: str = "",
    timestamp: datetime | None = None,
) -> Path:
    """Create and return ``<root>/<run_id>/`` (with a ``plots/`` subdir)."""
    run_dir = Path(root) / new_run_id(model_name, git_commit, timestamp)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


class _JSONEncoder(json.JSONEncoder):
    """Serialize dataclasses, Paths, and numpy scalars/arrays."""

    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        # numpy scalars expose .item(); arrays expose .tolist().
        item = getattr(o, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                pass
        tolist = getattr(o, "tolist", None)
        if callable(tolist):
            return tolist()
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, cls=_JSONEncoder)


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
