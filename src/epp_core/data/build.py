"""Deterministic raw-records → processed-reactions builder (dataset-agnostic).

Each dataset's ``prepare.py`` reads its raw files into record dicts (each with a
``reaction`` field holding a ``reactants>>products`` SMILES, plus any metadata to
carry) and calls :func:`build_reactions`. Splitting on ``>>``, per-molecule
canonicalization, identity/validity/length filtering, and deduplication live
here so they're consistent and unit-tested. Train/valid/test assignment is a
separate step (:mod:`epp_core.data.split`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

import pandas as pd

from epp_core.chem.reactions import (
    canonical_molecules,
    join_molecules,
    reaction_key,
    split_reaction,
)
from epp_core.chem.tokenize import atom_tokenize
from epp_core.data.schema import REACTION_COLUMNS, BuildStats

_DATA_COLUMNS = [c for c in REACTION_COLUMNS if c != "reaction_id"]


def build_reactions(
    records: Iterable[dict],
    *,
    max_tokens: int = 300,
    source: str = "",
    drop_identity: bool = True,
    dedup_extra_cols: Sequence[str] = (),
    extra_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, BuildStats]:
    """Build a deterministic processed-reactions frame and aggregate stats.

    Per record: split ``reaction`` into reactant/product molecule lists (drop if
    it isn't a single ``>>`` reaction), canonicalize every molecule (drop if any
    is invalid), optionally drop identity reactions (reactants == products), drop
    reactions whose atom-token length on either side exceeds ``max_tokens``, then
    dedupe on the canonical reaction key (plus ``dedup_extra_cols`` if given, so
    e.g. the same reaction under two EC numbers is kept as distinct rows). Output
    is sorted and assigned a stable ``reaction_id``, so rebuilding identical input
    yields an identical frame.

    ``extra_cols`` names per-record fields carried through verbatim (appended
    after :data:`REACTION_COLUMNS`); on a dedup collision the first-seen record's
    values win. ``dedup_extra_cols`` must be a subset of ``extra_cols``.
    """
    overlap = set(extra_cols) & set(REACTION_COLUMNS)
    if overlap:
        raise ValueError(f"extra_cols must not overlap REACTION_COLUMNS: {sorted(overlap)}")
    missing = set(dedup_extra_cols) - set(extra_cols)
    if missing:
        raise ValueError(f"dedup_extra_cols must be a subset of extra_cols: {sorted(missing)}")

    counter = {
        "n_in": 0,
        "n_invalid": 0,
        "n_identity": 0,
        "n_too_long": 0,
        "n_duplicate": 0,
        "n_out": 0,
    }
    seen: set[tuple] = set()
    rows: list[dict] = []

    for rec in records:
        counter["n_in"] += 1

        parsed = split_reaction(rec["reaction"])
        if parsed is None:
            counter["n_invalid"] += 1
            continue
        reactants = canonical_molecules(parsed[0])
        products = canonical_molecules(parsed[1])
        if reactants is None or products is None:
            counter["n_invalid"] += 1
            continue

        reactant_smiles = join_molecules(reactants)
        product_smiles = join_molecules(products)
        if drop_identity and reactant_smiles == product_smiles:
            counter["n_identity"] += 1
            continue

        src_tokens = atom_tokenize(reactant_smiles)
        tgt_tokens = atom_tokenize(product_smiles)
        if len(src_tokens) > max_tokens or len(tgt_tokens) > max_tokens:
            counter["n_too_long"] += 1
            continue

        key = (reaction_key(reactants, products), *(rec.get(c, "") for c in dedup_extra_cols))
        if key in seen:
            counter["n_duplicate"] += 1
            continue
        seen.add(key)

        row = {
            "reactant_smiles": reactant_smiles,
            "product_smiles": product_smiles,
            "source": source,
            "n_reactants": len(reactants),
            "n_products": len(products),
            "src_n_tokens": len(src_tokens),
            "tgt_n_tokens": len(tgt_tokens),
        }
        for col in extra_cols:
            row[col] = rec.get(col, "")
        rows.append(row)
        counter["n_out"] += 1

    sort_cols = ["reactant_smiles", "product_smiles", *dedup_extra_cols]
    df = pd.DataFrame(rows, columns=[*_DATA_COLUMNS, *extra_cols])
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.insert(0, "reaction_id", df.index.astype("int64"))
    df = df[[*REACTION_COLUMNS, *extra_cols]]

    return cast(pd.DataFrame, df), BuildStats(**counter)
