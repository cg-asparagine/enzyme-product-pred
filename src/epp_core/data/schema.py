"""Processed reaction-dataset schema shared by all reaction datasets."""

from __future__ import annotations

from dataclasses import dataclass

#: Columns produced by :func:`epp_core.data.build.build_reactions` (before split
#: assignment). The processed ``reactions.parquet`` additionally carries a
#: ``split`` column and any dataset-specific extra columns (e.g. ``ec_num``,
#: ``organism``, ``rxn_idx``).
REACTION_COLUMNS: list[str] = [
    "reaction_id",
    "reactant_smiles",
    "product_smiles",
    "source",
    "n_reactants",
    "n_products",
    "src_n_tokens",
    "tgt_n_tokens",
]

VALID_SPLITS: tuple[str, ...] = ("train", "valid", "test")


@dataclass
class BuildStats:
    """Aggregate counts recorded while building a processed reactions dataset."""

    n_in: int
    n_invalid: int
    n_identity: int
    n_too_long: int
    n_duplicate: int
    n_out: int
