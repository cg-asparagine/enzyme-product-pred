"""Build Victorien-CYP-metabolites: drug parent->metabolite pairs conditioned on the
metabolizing cytochrome-P450 enzyme.

This is an **evaluation-only** benchmark (all rows ``split="test"``) for the
sequence-conditioned reaction model. It is the *experimental-only* test split of
the ``Victorien-zaretzki-drugbank-v1`` dataset used to evaluate Metatrans-ReactionT5
in the sibling ``metabolite-prediction`` repo, re-cast as enzymatic reactions:
each parent->metabolite pair carries the human CYP isoform that produced it (the
``cyp`` column), which we map to that isoform's canonical reviewed UniProt entry
(``CYP_TO_ACCESSION``) and attach its amino-acid sequence (fetched from UniProt) so
the model can condition on the enzyme.

Raw input (``data/*/raw/`` is git-ignored; provenance is recorded here + in
``build_manifest.json``):
  * ``raw/metabolite_test_pairs.csv`` — the 233 experimental test pairs exported
    from the sibling repo (columns: pair_id, parent_smiles, metabolite_smiles, cyp,
    origin). Regenerate with, from the metabolite-prediction repo:
        import pandas as pd
        df = pd.read_parquet("data/Victorien-zaretzki-drugbank-v1/processed/pairs.parquet")
        (df[df["split"] == "test"][["pair_id", "parent_smiles", "metabolite_smiles",
            "cyp", "origin"]].to_csv(".../raw/metabolite_test_pairs.csv", index=False))
  * ``raw/uniprot_sequences.json`` — sequences fetched once from the UniProt REST API
    (cached, resumable); regenerated automatically by this script.

Rows whose ``cyp`` is not one of the 9 mapped isoforms (``other``/``unknown``/
``CYP_inferred``/``CYP_unspecified``) have no specific enzyme to condition on and
are dropped. Run with ``just data-prep Victorien-CYP-metabolites``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

import pandas as pd

from epp_core.data import build_reactions, content_hash, fetch_sequences, sha256_file
from epp_core.io import write_json

HERE = Path(__file__).resolve().parent
RAW_PAIRS = HERE / "raw" / "metabolite_test_pairs.csv"
SEQ_CACHE = HERE / "raw" / "uniprot_sequences.json"
PROCESSED = HERE / "processed"

DATASET_ID = "victorien-cyp-metabolites-v1"
SOURCE = "victorien-zaretzki-drugbank"
# All 9 mapped CYPs are EC 1.14.14.1 (unspecific monooxygenase); carried for provenance.
CYP_EC = "1.14.14.1"
ORGANISM = "Homo sapiens"
MAX_TOKENS = 300

# Human CYP isoform -> its canonical reviewed (Swiss-Prot) UniProt accession. Isoforms
# outside this map (other/unknown/CYP_inferred/CYP_unspecified) have no specific enzyme.
CYP_TO_ACCESSION = {
    "CYP1A2": "P05177",
    "CYP2A6": "P11509",
    "CYP2B6": "P20813",
    "CYP2C8": "P10632",
    "CYP2C9": "P11712",
    "CYP2C19": "P33261",
    "CYP2D6": "P10635",
    "CYP2E1": "P05181",
    "CYP3A4": "P08684",
}

EXTRA_COLS = ["uniprot_id", "cyp", "ec_num", "organism", "origin", "source_pair_id"]
# Keep the same parent->metabolite reaction under two different CYPs as distinct rows.
DEDUP_EXTRA_COLS = ["uniprot_id"]


def _to_int(value: object) -> int:
    """Coerce a pandas/numpy scalar reduction result to a plain int."""
    return int(value)  # type: ignore[arg-type]  # pandas-stubs widen reductions to Series | int


def load_records() -> tuple[list[dict], int, dict[str, int], int]:
    """Read the test pairs into per-pair reaction records, mapping cyp -> accession.

    Returns the records, the raw row count, the per-CYP kept counts, and the number
    of rows dropped for having no mappable (specific) CYP isoform.
    """
    pairs = pd.read_csv(RAW_PAIRS)
    n_raw = len(pairs)

    records: list[dict] = []
    kept_by_cyp: dict[str, int] = {}
    n_dropped_unmapped = 0
    for pair_id, parent, metabolite, cyp, origin in zip(
        pairs["pair_id"],
        pairs["parent_smiles"],
        pairs["metabolite_smiles"],
        pairs["cyp"],
        pairs["origin"],
        strict=True,
    ):
        accession = CYP_TO_ACCESSION.get(str(cyp))
        if accession is None:  # other / unknown / CYP_inferred / CYP_unspecified
            n_dropped_unmapped += 1
            continue
        kept_by_cyp[str(cyp)] = kept_by_cyp.get(str(cyp), 0) + 1
        records.append(
            {
                "reaction": f"{parent}>>{metabolite}",
                "uniprot_id": accession,
                "cyp": str(cyp),
                "ec_num": CYP_EC,
                "organism": ORGANISM,
                "origin": str(origin),
                "source_pair_id": int(pair_id),
            }
        )
    return records, n_raw, kept_by_cyp, n_dropped_unmapped


def main() -> None:
    if not RAW_PAIRS.exists():
        raise FileNotFoundError(
            f"{RAW_PAIRS} not found — see data/Victorien-CYP-metabolites/README.md for how "
            f"to export it from the metabolite-prediction repo."
        )
    PROCESSED.mkdir(exist_ok=True)

    records, n_raw, kept_by_cyp, n_dropped_unmapped = load_records()
    df, stats = build_reactions(
        records,
        max_tokens=MAX_TOKENS,
        source=SOURCE,
        extra_cols=EXTRA_COLS,
        dedup_extra_cols=DEDUP_EXTRA_COLS,
    )
    n_pairs = len(df)

    # Fetch + attach the enzyme sequence per accession (cached in raw/, resumable).
    accessions = sorted(set(df["uniprot_id"]))
    print(f"Fetching sequences for {len(accessions)} CYP accessions (cached)...")
    sequences = fetch_sequences(accessions, cache_path=SEQ_CACHE)
    df["sequence"] = df["uniprot_id"].map(lambda a: sequences.get(str(a), ""))
    df = cast(pd.DataFrame, df[df["sequence"] != ""].reset_index(drop=True))
    n_dropped_no_seq = n_pairs - len(df)
    df["seq_len"] = df["sequence"].str.len()
    df["reaction_id"] = range(len(df))  # re-contiguous after any drop
    df["split"] = "test"  # evaluation-only benchmark

    df.to_parquet(PROCESSED / "reactions.parquet", index=False)

    manifest = {
        "dataset_id": DATASET_ID,
        "source": SOURCE,
        "description": "Experimental-only test split of Victorien-zaretzki-drugbank-v1 "
        "(metabolite-prediction repo), re-cast as CYP-conditioned reactions.",
        "content_hash": content_hash(df),
        "raw_files": {"metabolite_test_pairs.csv": sha256_file(RAW_PAIRS)},
        "n_raw_pairs": n_raw,
        "n_dropped_unmapped_cyp": n_dropped_unmapped,
        "n_dropped_no_seq": n_dropped_no_seq,
        "n_out": len(df),
        "split": "all rows are test (evaluation-only)",
        "max_tokens": MAX_TOKENS,
        "dedup_extra_cols": DEDUP_EXTRA_COLS,
        "build_stats": asdict(stats),
        "cyp_to_uniprot": dict(sorted(CYP_TO_ACCESSION.items())),
        "kept_by_cyp": dict(sorted(kept_by_cyp.items())),
        "n_enzymes": _to_int(df["uniprot_id"].nunique()),
    }
    write_json(PROCESSED / "build_manifest.json", manifest)

    kept = dict(sorted(kept_by_cyp.items()))
    print(f"Built {len(df)} CYP-conditioned reactions from {n_raw} raw pairs -> {PROCESSED}")
    print(f"  dropped {n_dropped_unmapped} pairs with no specific CYP isoform")
    print(f"  enzymes: {df['uniprot_id'].nunique()} CYPs; kept_by_cyp: {kept}")


if __name__ == "__main__":
    main()
