"""Tests for epp_core.io: run-id naming and robust JSON (de)serialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from epp_core.io import new_run_dir, new_run_id, read_json, write_json


def test_write_read_json_roundtrip(tmp_path: Path) -> None:
    obj = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    path = tmp_path / "sub" / "data.json"
    write_json(path, obj)
    assert path.exists()
    assert read_json(path) == obj


@dataclass
class _Sample:
    x: int
    y: str


def test_write_json_serializes_dataclasses_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    write_json(path, {"sample": _Sample(x=1, y="hi"), "path": Path("/tmp/foo")})
    loaded = read_json(path)
    assert loaded["sample"] == {"x": 1, "y": "hi"}
    assert loaded["path"] == "/tmp/foo"


def test_new_run_id_is_sortable_and_slugged() -> None:
    ts = datetime(2026, 6, 19, 15, 4, 30, tzinfo=UTC)
    run_id = new_run_id("EC Lookup/Baseline", "abc1234567", timestamp=ts)
    assert run_id == "20260619T150430Z__EC-Lookup-Baseline__abc1234"


def test_new_run_id_handles_missing_git() -> None:
    run_id = new_run_id("m", timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    assert run_id.endswith("__nogit")


def test_new_run_dir_creates_plots_subdir(tmp_path: Path) -> None:
    run_dir = new_run_dir(tmp_path, "model", "deadbeef", timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    assert run_dir.is_dir()
    assert (run_dir / "plots").is_dir()
