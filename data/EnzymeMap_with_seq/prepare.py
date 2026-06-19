"""Build EnzymeMap_with_seq: EnzymeMap reactions that carry a UniProt enzyme sequence.

Like EnzymeMap, but restricted to rows with a UniProt accession
(``protein_db`` in {uniprot, swissprot}); the protein sequence for each accession
is fetched from UniProt and attached as a ``sequence`` column. An example is a
unique (canonical reaction, UniProt ID) pair. Run with
``just data-prep EnzymeMap_with_seq``.

Reuses the EnzymeMap raw CSV (single source of truth). The fetched sequences are
cached under ``raw/uniprot_sequences.json`` (git-ignored); ``processed/`` is
git-ignored too.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pandas as pd

from epp_core.chem.reactions import undirected_reaction_key
from epp_core.data import (
    assign_splits,
    build_reactions,
    content_hash,
    fetch_sequences,
    sha256_file,
)
from epp_core.data.schema import VALID_SPLITS
from epp_core.io import write_json

HERE = Path(__file__).resolve().parent
# Reuse EnzymeMap's raw CSV — one source of truth for the raw data.
RAW = HERE.parent / "EnzymeMap" / "raw" / "enzymemap_v2_brenda2023.csv"
PROCESSED = HERE / "processed"
SEQ_CACHE = HERE / "raw" / "uniprot_sequences.json"

DATASET_ID = "enzymemap-with-seq-v1"
SOURCE = "enzymemap-v2-brenda2023"

# --- Filters (documented in README) ---
ONLY_SINGLE_STEP = True
MIN_QUALITY = 0.3
MAX_TOKENS = 300
UNIPROT_DBS = ("uniprot", "swissprot")  # both are UniProtKB accessions; genbank excluded

# --- Split (grouped on the direction-collapsed reaction key) ---
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42

EXTRA_COLS = [
    "rxn_idx",
    "ec_num",
    "organism",
    "direction",
    "quality",
    "natural",
    "steps",
    "uniprot_id",
]
# One example per unique (canonical reaction, UniProt ID).
DEDUP_EXTRA_COLS = ["uniprot_id"]


def _direction(source: str) -> str:
    return "reversed" if "reversed" in str(source) else "forward"


def _to_int(value: object) -> int:
    """Coerce a pandas/numpy scalar reduction result to a plain int."""
    return int(value)  # type: ignore[arg-type]  # pandas-stubs widen reductions to Series | int


def _parse_refs(value: object) -> list[str]:
    """Parse a ``protein_refs`` cell (e.g. ``"['P12345', 'Q67890']"``) into accessions."""
    try:
        parsed = ast.literal_eval(value)  # type: ignore[arg-type]
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(a) for a in parsed]
    return [str(parsed)]


def load_records() -> tuple[list[dict], int]:
    """Read + filter the raw CSV into per-(reaction, accession) record dicts."""
    raw = pd.read_csv(RAW)
    n_raw = len(raw)
    mask = raw["quality"] >= MIN_QUALITY
    if ONLY_SINGLE_STEP:
        mask = mask & (raw["steps"] == "single")
    mask = mask & raw["protein_db"].isin(UNIPROT_DBS)
    df = cast(pd.DataFrame, raw[mask])

    records: list[dict] = []
    for unmapped, rxn_idx, ec_num, organism, source, quality, natural, steps, refs in zip(
        df["unmapped"],
        df["rxn_idx"],
        df["ec_num"],
        df["organism"],
        df["source"],
        df["quality"],
        df["natural"],
        df["steps"],
        df["protein_refs"],
        strict=True,
    ):
        for accession in _parse_refs(refs):  # explode multi-accession rows
            records.append(
                {
                    "reaction": unmapped,
                    "rxn_idx": int(rxn_idx),
                    "ec_num": str(ec_num),
                    "organism": "" if pd.isna(organism) else str(organism),
                    "direction": _direction(source),
                    "quality": float(quality),
                    "natural": bool(natural),
                    "steps": str(steps),
                    "uniprot_id": accession,
                }
            )
    return records, n_raw


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. EnzymeMap_with_seq reuses EnzymeMap's raw CSV — "
            f"see data/EnzymeMap/README.md."
        )
    PROCESSED.mkdir(exist_ok=True)

    records, n_raw = load_records()
    df, stats = build_reactions(
        records,
        max_tokens=MAX_TOKENS,
        source=SOURCE,
        extra_cols=EXTRA_COLS,
        dedup_extra_cols=DEDUP_EXTRA_COLS,
    )
    n_pairs = len(df)

    # Fetch + attach sequences; drop pairs whose accession didn't resolve.
    accessions = sorted(set(df["uniprot_id"]))
    print(f"Fetching sequences for {len(accessions)} unique UniProt accessions (cached)...")
    sequences = fetch_sequences(accessions, cache_path=SEQ_CACHE)
    df["sequence"] = df["uniprot_id"].map(lambda a: sequences.get(a, ""))
    df = cast(pd.DataFrame, df[df["sequence"] != ""].reset_index(drop=True))
    df["seq_len"] = df["sequence"].str.len()
    n_dropped_no_seq = n_pairs - len(df)
    df["reaction_id"] = range(len(df))  # re-contiguous after the drop

    # Group splits on the direction-collapsed reaction identity (no forward/reverse leak).
    df["_undirected"] = df.apply(
        lambda r: undirected_reaction_key(
            r["reactant_smiles"].split("."), r["product_smiles"].split(".")
        ),
        axis=1,
    )
    df = assign_splits(df, group_col="_undirected", fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED)
    spans = df.groupby("_undirected")["split"].nunique()
    if _to_int((spans > 1).sum()):
        raise SystemExit("LEAKAGE DETECTED: a reaction identity spans multiple splits")
    df = df.drop(columns=["_undirected"])

    df.to_parquet(PROCESSED / "reactions.parquet", index=False)

    per_split = {s: _to_int((df["split"] == s).sum()) for s in VALID_SPLITS}
    manifest = {
        "dataset_id": DATASET_ID,
        "source": SOURCE,
        "content_hash": content_hash(df),
        "raw_file": RAW.name,
        "raw_sha256": sha256_file(RAW),
        "n_raw_rows": n_raw,
        "filters": {
            "only_single_step": ONLY_SINGLE_STEP,
            "min_quality": MIN_QUALITY,
            "max_tokens": MAX_TOKENS,
            "uniprot_dbs": list(UNIPROT_DBS),
        },
        "split": {
            "fractions": list(SPLIT_FRACTIONS),
            "seed": SPLIT_SEED,
            "group": "undirected_reaction_key",
        },
        "dedup_extra_cols": DEDUP_EXTRA_COLS,
        "build_stats": asdict(stats),
        "n_pairs_before_seq": n_pairs,
        "n_unique_accessions": len(accessions),
        "n_accessions_resolved": len(sequences),
        "n_dropped_no_seq": n_dropped_no_seq,
        "n_out": len(df),
        "per_split": per_split,
        "n_ec_numbers": _to_int(df["ec_num"].nunique()),
    }
    write_json(PROCESSED / "build_manifest.json", manifest)

    print(f"Built {len(df)} (reaction, protein) examples from {n_raw} raw rows -> {PROCESSED}")
    print(f"  per split: {per_split}")
    print(
        f"  accessions: {len(sequences)}/{len(accessions)} resolved, "
        f"{n_dropped_no_seq} pairs dropped (no sequence)"
    )


if __name__ == "__main__":
    main()
