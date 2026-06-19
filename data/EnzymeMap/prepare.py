"""Build the processed EnzymeMap dataset from the raw EnzymeMap v2 / BRENDA 2023 CSV.

Run with ``just data-prep EnzymeMap``. Reads the raw CSV, applies the filters
below, canonicalizes each reaction via :func:`epp_core.data.build_reactions`,
assigns a grouped-random train/valid/test split (grouped on the
direction-collapsed reaction identity so forward/reverse twins never leak across
splits), and writes ``processed/reactions.parquet`` + ``build_manifest.json``.
The raw CSV (~386 MB) is git-ignored under ``raw/``; see README.md for provenance.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from epp_core.chem.reactions import undirected_reaction_key
from epp_core.data import assign_splits, build_reactions, content_hash, sha256_file
from epp_core.data.schema import VALID_SPLITS
from epp_core.io import write_json

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "enzymemap_v2_brenda2023.csv"
PROCESSED = HERE / "processed"

DATASET_ID = "enzymemap-brenda2023-v1"
SOURCE = "enzymemap-v2-brenda2023"

# --- Filters (documented in README; edit here to rebuild a different cut) ---
ONLY_SINGLE_STEP = True  # keep steps == "single" (drop multi / single-from-multi)
MIN_QUALITY = 0.3  # drop the low-confidence tail
MAX_TOKENS = 300  # atom-token cap per reaction side (cofactors make sides long)

# --- Split (grouped on the direction-collapsed reaction key; see CLAUDE.md roadmap) ---
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42

# Metadata carried into the processed table for conditioning / analysis.
EXTRA_COLS = ["rxn_idx", "ec_num", "organism", "direction", "quality", "natural", "steps"]
# The same reaction under different EC numbers is a distinct training signal.
DEDUP_EXTRA_COLS = ["ec_num"]


def _direction(source: str) -> str:
    return "reversed" if "reversed" in str(source) else "forward"


def _to_int(value: object) -> int:
    """Coerce a pandas/numpy scalar reduction result to a plain int."""
    return int(value)  # type: ignore[arg-type]  # pandas-stubs widen reductions to Series | int


def load_records() -> tuple[list[dict], int]:
    """Read + filter the raw CSV into ``build_reactions`` record dicts."""
    df = pd.read_csv(RAW)
    n_raw = len(df)
    if ONLY_SINGLE_STEP:
        df = df[df["steps"] == "single"]
    df = df[df["quality"] >= MIN_QUALITY]

    records: list[dict] = []
    for unmapped, rxn_idx, ec_num, organism, source, quality, natural, steps in zip(
        df["unmapped"],
        df["rxn_idx"],
        df["ec_num"],
        df["organism"],
        df["source"],
        df["quality"],
        df["natural"],
        df["steps"],
        strict=True,
    ):
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
            }
        )
    return records, n_raw


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found. See {HERE / 'README.md'} for how to obtain it.")
    PROCESSED.mkdir(exist_ok=True)

    records, n_raw = load_records()
    df, stats = build_reactions(
        records,
        max_tokens=MAX_TOKENS,
        source=SOURCE,
        extra_cols=EXTRA_COLS,
        dedup_extra_cols=DEDUP_EXTRA_COLS,
    )

    # Group splits on the direction-collapsed reaction identity so forward/reverse
    # twins (and the same reaction across organisms) never straddle a split.
    df["_undirected"] = df.apply(
        lambda r: undirected_reaction_key(
            r["reactant_smiles"].split("."), r["product_smiles"].split(".")
        ),
        axis=1,
    )
    df = assign_splits(df, group_col="_undirected", fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED)

    # Leakage check (passes by construction; assert anyway as a guardrail).
    spans = df.groupby("_undirected")["split"].nunique()
    leaked = _to_int((spans > 1).sum())
    if leaked:
        raise SystemExit(f"LEAKAGE DETECTED: {leaked} reaction identities span multiple splits")
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
        },
        "split": {
            "fractions": list(SPLIT_FRACTIONS),
            "seed": SPLIT_SEED,
            "group": "undirected_reaction_key",
        },
        "dedup_extra_cols": DEDUP_EXTRA_COLS,
        "build_stats": asdict(stats),
        "n_out": len(df),
        "per_split": per_split,
        "n_ec_numbers": _to_int(df["ec_num"].nunique()),
    }
    write_json(PROCESSED / "build_manifest.json", manifest)

    print(f"Built {len(df)} reactions from {n_raw} raw rows -> {PROCESSED}")
    print(f"  per split: {per_split}")
    print(
        f"  EC numbers: {manifest['n_ec_numbers']}  content_hash: {manifest['content_hash'][:12]}"
    )


if __name__ == "__main__":
    main()
