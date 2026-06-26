"""Build EnzymeMap_with_seq_plus: a superset of EnzymeMap_with_seq that also includes
reactions whose enzyme had no curated UniProt accession, recovered by resolving the
reaction's (EC number, organism) to candidate UniProtKB accessions.

Two provenances, distinguished by the ``accession_source`` column:
- ``curated``  — the accession came from the raw CSV's ``protein_refs`` (a direct
  BRENDA -> UniProt link), exactly as in EnzymeMap_with_seq.
- ``resolved`` — the row had no accession; its (EC, organism) was looked up in
  ``resolved_accessions.csv`` (produced by ``resolve_accessions.py`` via
  ``epp_core.data.search_accessions``) and the best-reviewed candidate attached.

Resolved accessions are candidate enzymes of the right (EC, organism) — correct
activity + species — but are NOT verified to catalyze the specific reaction. Filter
on ``accession_source == "curated"`` for a high-precision cut; use ``resolved`` rows
to expand training coverage. See README.md.

Reuses EnzymeMap's raw CSV and EnzymeMap_with_seq's sequence caches + resolved table
(one source of truth each). ``processed/`` is git-ignored. Run with
``just data-prep EnzymeMap_with_seq_plus``.
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
    cluster_sequences,
    content_hash,
    fetch_sequences,
    sha256_file,
    uniparc_sequences,
)
from epp_core.data.schema import VALID_SPLITS
from epp_core.io import write_json

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "EnzymeMap_with_seq"
# Reuse EnzymeMap's raw CSV + EnzymeMap_with_seq's caches/resolved table (one source
# of truth each) so nothing is re-fetched or recomputed.
RAW = HERE.parent / "EnzymeMap" / "raw" / "enzymemap_v2_brenda2023.csv"
PROCESSED = HERE / "processed"
SEQ_CACHE = SIBLING / "raw" / "uniprot_sequences.json"
UNIPARC_CACHE = SIBLING / "raw" / "uniparc_sequences.json"
RESOLVED_TABLE = SIBLING / "processed" / "resolved_accessions.csv"

DATASET_ID = "enzymemap-with-seq-plus-v1"
SOURCE = "enzymemap-v2-brenda2023"

# --- Filters (must match EnzymeMap_with_seq + how resolved_accessions.csv was built) ---
ONLY_SINGLE_STEP = True
MIN_QUALITY = 0.3
MAX_TOKENS = 300
UNIPROT_DBS = ("uniprot", "swissprot")  # both are UniProtKB accessions; genbank excluded

# When a resolved (EC, organism) maps to several candidate accessions:
#   "pick_one"    -> keep only the first (reviewed/Swiss-Prot-first) representative.
#   "explode_all" -> one example per candidate (<=5), maximizing sequence diversity.
MULTI_ACCESSION = "pick_one"

# --- Split (grouped on the direction-collapsed reaction key) ---
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42

# --- Enzyme-cluster split (sequence-similarity clustering of enzymes) ---
# Holds out whole enzyme clusters so test enzymes (and close homologs) are unseen in
# training — the honest generalization test for a sequence-conditioned model. k-mer
# Jaccard params match EnzymeMap_with_seq; the clusterer is inverted-index (not
# all-pairs) so it scales to the larger protein set. See epp_core.data.cluster.
ENZYME_CLUSTER_METHOD = "kmer"
ENZYME_CLUSTER_K = 4
ENZYME_CLUSTER_THRESHOLD = 0.4

EXTRA_COLS = [
    "rxn_idx",
    "ec_num",
    "organism",
    "direction",
    "quality",
    "natural",
    "steps",
    "uniprot_id",
    "accession_source",
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


def _has_accession(protein_db: object, protein_refs: object) -> bool:
    """True if a raw row already carries a curated UniProtKB accession."""
    return protein_db in UNIPROT_DBS and len(_parse_refs(protein_refs)) > 0


def _quality_mask(raw: pd.DataFrame) -> pd.Series:
    """The shared quality + single-step row filter (before the accession split)."""
    mask = raw["quality"] >= MIN_QUALITY
    if ONLY_SINGLE_STEP:
        mask = mask & (raw["steps"] == "single")
    return mask


def load_records(raw: pd.DataFrame) -> list[dict]:
    """Curated path: per-(reaction, curated-accession) records from rows that already
    carry a UniProt accession (``protein_db`` in {uniprot, swissprot})."""
    df = cast(pd.DataFrame, raw[_quality_mask(raw) & raw["protein_db"].isin(UNIPROT_DBS)])
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
                    "accession_source": "curated",
                }
            )
    return records


def load_resolved_records(
    raw: pd.DataFrame, resolved: pd.DataFrame, *, explode_all: bool
) -> list[dict]:
    """Resolved path: records for accession-LESS dataset-quality rows, attaching the
    candidate accessions looked up by (EC, organism) from the resolved table.

    The accession-less set is the exact complement of ``load_records`` (rows that
    failed ``_has_accession``), so curated and resolved source rows are disjoint.
    """
    lookup: dict[tuple[str, str], list[str]] = {}
    for ec_num, organism, accs in zip(
        resolved["ec_num"], resolved["organism"], resolved["accessions"], strict=True
    ):
        key = (str(ec_num), "" if pd.isna(organism) else str(organism))
        lookup[key] = [a for a in str(accs).split(";") if a]

    df = cast(pd.DataFrame, raw[_quality_mask(raw)])
    records: list[dict] = []
    for unmapped, rxn_idx, ec_num, organism, source, quality, natural, steps, db, refs in zip(
        df["unmapped"],
        df["rxn_idx"],
        df["ec_num"],
        df["organism"],
        df["source"],
        df["quality"],
        df["natural"],
        df["steps"],
        df["protein_db"],
        df["protein_refs"],
        strict=True,
    ):
        if _has_accession(db, refs):  # curated rows are handled by load_records()
            continue
        organism_str = "" if pd.isna(organism) else str(organism)
        if not organism_str:  # no organism -> can't have been resolved
            continue
        accessions = lookup.get((str(ec_num), organism_str), [])
        if not accessions:  # (EC, organism) wasn't resolved to any accession
            continue
        chosen = accessions if explode_all else accessions[:1]
        for accession in chosen:
            records.append(
                {
                    "reaction": unmapped,
                    "rxn_idx": int(rxn_idx),
                    "ec_num": str(ec_num),
                    "organism": organism_str,
                    "direction": _direction(source),
                    "quality": float(quality),
                    "natural": bool(natural),
                    "steps": str(steps),
                    "uniprot_id": accession,
                    "accession_source": "resolved",
                }
            )
    return records


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. EnzymeMap_with_seq_plus reuses EnzymeMap's raw CSV — "
            f"see data/EnzymeMap/README.md."
        )
    if not RESOLVED_TABLE.exists():
        raise FileNotFoundError(
            f"{RESOLVED_TABLE} not found. Run "
            f"`uv run python data/EnzymeMap_with_seq/resolve_accessions.py` first."
        )
    PROCESSED.mkdir(exist_ok=True)

    raw = pd.read_csv(RAW)
    n_raw = len(raw)
    resolved_table = pd.read_csv(RESOLVED_TABLE)

    # Curated records FIRST, then resolved: on a (reaction, uniprot_id) dedup collision
    # build_reactions keeps the first-seen, so curated wins (accession_source is not in
    # the dedup key).
    records = load_records(raw)
    records += load_resolved_records(
        raw, resolved_table, explode_all=(MULTI_ACCESSION == "explode_all")
    )

    df, stats = build_reactions(
        records,
        max_tokens=MAX_TOKENS,
        source=SOURCE,
        extra_cols=EXTRA_COLS,
        dedup_extra_cols=DEDUP_EXTRA_COLS,
    )
    n_pairs = len(df)

    # Fetch + attach sequences; drop pairs whose accession didn't resolve. The accession
    # set is curated ∪ resolved de-duped (a set), so shared accessions are fetched once
    # and the EnzymeMap_with_seq cache makes already-seen ones free.
    accessions = sorted(set(df["uniprot_id"]))
    print(f"Fetching sequences for {len(accessions)} unique UniProt accessions (cached)...")
    sequences = fetch_sequences(accessions, cache_path=SEQ_CACHE)
    n_resolved_direct = len(sequences)

    # Recover accessions the live endpoint dropped (obsolete/secondary) from the
    # UniParc archive — one lookup per missing accession (cached, resumable).
    missing = [a for a in accessions if a not in sequences]
    n_recovered_uniparc = 0
    if missing:
        print(f"Recovering {len(missing)} missing accessions from the UniParc archive...")
        recovered = uniparc_sequences(missing, cache_path=UNIPARC_CACHE)
        n_recovered_uniparc = len(recovered)
        sequences = {**sequences, **recovered}
        print(f"  recovered {n_recovered_uniparc} of {len(missing)} via UniParc")

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

    # Enzyme-cluster split: cluster enzymes by sequence similarity, then assign
    # whole clusters to train/valid/test so homologous enzymes never straddle.
    print(f"Clustering {df['uniprot_id'].nunique()} enzymes ({ENZYME_CLUSTER_METHOD})...")
    clusters = cluster_sequences(
        dict(zip(df["uniprot_id"], df["sequence"], strict=True)),
        method=ENZYME_CLUSTER_METHOD,
        k=ENZYME_CLUSTER_K,
        threshold=ENZYME_CLUSTER_THRESHOLD,
    )
    df["enzyme_cluster"] = df["uniprot_id"].map(lambda u: clusters[u])
    df = assign_splits(
        df,
        group_col="enzyme_cluster",
        fractions=SPLIT_FRACTIONS,
        seed=SPLIT_SEED,
        split_col="enzyme_split",
    )
    cluster_spans = df.groupby("enzyme_cluster")["enzyme_split"].nunique()
    if _to_int((cluster_spans > 1).sum()):
        raise SystemExit("LEAKAGE DETECTED: an enzyme cluster spans multiple splits")

    df.to_parquet(PROCESSED / "reactions.parquet", index=False)

    per_split = {s: _to_int((df["split"] == s).sum()) for s in VALID_SPLITS}
    n_curated = _to_int((df["accession_source"] == "curated").sum())
    n_resolved = _to_int((df["accession_source"] == "resolved").sum())
    n_distinct_resolved_acc = len(
        {a for cell in resolved_table["accessions"] for a in str(cell).split(";") if a}
    )
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
        "multi_accession": MULTI_ACCESSION,
        "accession_source_counts": {"curated": n_curated, "resolved": n_resolved},
        "resolved_input": {
            "table_file": RESOLVED_TABLE.name,
            "table_sha256": sha256_file(RESOLVED_TABLE),
            "n_pairs_resolved": len(resolved_table),
            "n_distinct_resolved_accessions": n_distinct_resolved_acc,
        },
        "split": {
            "fractions": list(SPLIT_FRACTIONS),
            "seed": SPLIT_SEED,
            "group": "undirected_reaction_key",
        },
        "enzyme_split": {
            "method": ENZYME_CLUSTER_METHOD,
            "k": ENZYME_CLUSTER_K,
            "threshold": ENZYME_CLUSTER_THRESHOLD,
            "fractions": list(SPLIT_FRACTIONS),
            "seed": SPLIT_SEED,
            "group": "enzyme_cluster",
            "n_clusters": _to_int(df["enzyme_cluster"].nunique()),
            "per_split": {s: _to_int((df["enzyme_split"] == s).sum()) for s in VALID_SPLITS},
        },
        "dedup_extra_cols": DEDUP_EXTRA_COLS,
        "build_stats": asdict(stats),
        "n_pairs_before_seq": n_pairs,
        "n_unique_accessions": len(accessions),
        "n_resolved_direct": n_resolved_direct,
        "n_recovered_uniparc": n_recovered_uniparc,
        "n_accessions_resolved": len(sequences),
        "n_dropped_no_seq": n_dropped_no_seq,
        "n_out": len(df),
        "per_split": per_split,
        "n_ec_numbers": _to_int(df["ec_num"].nunique()),
    }
    write_json(PROCESSED / "build_manifest.json", manifest)

    print(f"Built {len(df)} (reaction, protein) examples from {n_raw} raw rows -> {PROCESSED}")
    print(f"  provenance: {n_curated} curated, {n_resolved} resolved")
    print(f"  per split: {per_split}")
    print(
        f"  accessions: {len(sequences)}/{len(accessions)} resolved, "
        f"{n_dropped_no_seq} pairs dropped (no sequence)"
    )


if __name__ == "__main__":
    main()
