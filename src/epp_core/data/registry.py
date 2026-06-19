"""Content hashing for dataset reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    """A stable hash of a processed-reactions frame, independent of ``reaction_id``/row order."""
    columns = [str(c) for c in df.columns if c != "reaction_id"]
    frame = cast(pd.DataFrame, df[columns]).sort_values(columns)
    canonical = frame.to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
