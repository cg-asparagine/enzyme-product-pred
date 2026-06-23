"""Load processed reaction datasets for models and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def load_reactions(
    processed_dir: str | Path, split: str | None = None, split_col: str = "split"
) -> pd.DataFrame:
    """Load ``reactions.parquet`` from a dataset's ``processed/`` dir, optionally
    filtered to one split.

    ``split_col`` selects which split column to filter on (default ``"split"``, the
    reaction-grouped split); pass e.g. ``"enzyme_split"`` to use the enzyme-cluster
    split for an honest new-enzyme generalization test.
    """
    df = pd.read_parquet(Path(processed_dir) / "reactions.parquet")
    if split is not None:
        df = df[df[split_col] == split].reset_index(drop=True)
    return cast(pd.DataFrame, df)


def train_product_smiles(processed_dir: str | Path, split_col: str = "split") -> set[str]:
    """Canonical product-molecule SMILES seen in training (for the novelty metric)."""
    smiles: set[str] = set()
    for product_smiles in load_reactions(processed_dir, "train", split_col)["product_smiles"]:
        smiles.update(product_smiles.split("."))
    return smiles
