"""Load processed reaction datasets for models and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def load_reactions(processed_dir: str | Path, split: str | None = None) -> pd.DataFrame:
    """Load ``reactions.parquet`` from a dataset's ``processed/`` dir, optionally
    filtered to one split."""
    df = pd.read_parquet(Path(processed_dir) / "reactions.parquet")
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return cast(pd.DataFrame, df)


def train_product_smiles(processed_dir: str | Path) -> set[str]:
    """Canonical product-molecule SMILES seen in training (for the novelty metric)."""
    smiles: set[str] = set()
    for product_smiles in load_reactions(processed_dir, "train")["product_smiles"]:
        smiles.update(product_smiles.split("."))
    return smiles
