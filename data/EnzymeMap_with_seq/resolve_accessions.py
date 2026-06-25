"""Resolve UniProt accessions for EnzymeMap reactions that lack one.

Most EnzymeMap rows carry an EC number + organism but no UniProt accession
(``protein_db`` is null or ``genbank``), so they're dropped from
``EnzymeMap_with_seq``. This script takes every such ``(EC, organism)`` pair from
the dataset-quality subset (``quality >= 0.3``, single-step) and resolves it to
candidate UniProtKB accessions via ``epp_core.data.search_accessions`` (strict:
exact organism, reviewed/Swiss-Prot entries first).

Resumable: lookups are cached to ``raw/ec_organism_accessions.json`` (git-ignored,
written every few lookups), so an interrupted run continues where it left off and
re-runs are instant. On completion it writes a resolved table to
``processed/resolved_accessions.csv`` and prints coverage. Feed the accessions into
``epp_core.data.fetch_sequences`` to attach sequences (as ``prepare.py`` does).

Run: ``uv run python data/EnzymeMap_with_seq/resolve_accessions.py``
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pandas as pd

from epp_core.data import search_accessions

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "EnzymeMap" / "raw" / "enzymemap_v2_brenda2023.csv"
ACC_CACHE = HERE / "raw" / "ec_organism_accessions.json"
OUT_TABLE = HERE / "processed" / "resolved_accessions.csv"

# Same reaction filters as the dataset builds (prepare.py).
MIN_QUALITY = 0.3
ONLY_SINGLE_STEP = True
UNIPROT_DBS = ("uniprot", "swissprot")

# Resolver settings (strict exact-organism match; Swiss-Prot first).
MAX_PER_QUERY = 5
PREFER_REVIEWED = True
SLEEP = 0.1


def _has_accession(protein_db: object, protein_refs: object) -> bool:
    """True if this raw row already carries a UniProtKB accession."""
    if protein_db not in UNIPROT_DBS:
        return False
    try:
        parsed = ast.literal_eval(cast(str, protein_refs))
    except (ValueError, SyntaxError):
        return False
    return isinstance(parsed, (list, tuple)) and len(parsed) > 0


def accession_less_pairs() -> pd.DataFrame:
    """Unique ``(ec_num, organism)`` pairs from accession-less, dataset-quality rows."""
    raw = pd.read_csv(RAW)
    mask = raw["quality"] >= MIN_QUALITY
    if ONLY_SINGLE_STEP:
        mask = mask & (raw["steps"] == "single")
    df = cast(pd.DataFrame, raw[mask]).copy()
    df["_has_acc"] = [
        _has_accession(db, refs)
        for db, refs in zip(df["protein_db"], df["protein_refs"], strict=True)
    ]
    df["ec_num"] = df["ec_num"].astype(str)
    df["organism"] = df["organism"].fillna("").astype(str)
    miss = df[(~df["_has_acc"]) & (df["organism"] != "")]
    pairs = (
        cast(pd.DataFrame, miss[["ec_num", "organism"]])
        .drop_duplicates()
        .sort_values(["ec_num", "organism"])
        .reset_index(drop=True)
    )
    return pairs


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found — see data/EnzymeMap/README.md.")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    pairs = accession_less_pairs()
    queries = list(zip(pairs["ec_num"], pairs["organism"], strict=True))
    print(
        f"Resolving {len(queries)} unique (EC, organism) pairs "
        f"({pairs['ec_num'].nunique()} ECs, {pairs['organism'].nunique()} organisms)..."
    )
    print(f"  cache: {ACC_CACHE}  (resumable; re-run to continue)")

    resolved = search_accessions(
        queries,
        cache_path=ACC_CACHE,
        max_per_query=MAX_PER_QUERY,
        prefer_reviewed=PREFER_REVIEWED,
        sleep=SLEEP,
    )

    rows = [
        {
            "ec_num": ec,
            "organism": organism,
            "n_accessions": len(resolved[(ec, organism)]),
            "accessions": ";".join(resolved[(ec, organism)]),
        }
        for (ec, organism) in queries
        if (ec, organism) in resolved
    ]
    out = pd.DataFrame(rows, columns=["ec_num", "organism", "n_accessions", "accessions"])
    out.to_csv(OUT_TABLE, index=False)

    n_acc = int(cast(int, out["n_accessions"].sum()))
    n_distinct = len({a for v in resolved.values() for a in v})
    print(f"\nResolved {len(out)}/{len(queries)} pairs ({100 * len(out) / len(queries):.1f}%)")
    print(f"  ECs with >=1 accession: {out['ec_num'].nunique()}/{pairs['ec_num'].nunique()}")
    print(f"  accessions found: {n_acc} ({n_distinct} distinct)")
    print(f"  table -> {OUT_TABLE}")


if __name__ == "__main__":
    main()
