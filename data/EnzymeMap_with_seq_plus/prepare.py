"""Build EnzymeMap_with_seq_plus: a superset of EnzymeMap_with_seq that also includes
reactions whose enzyme had no curated UniProt accession (recovered by resolving the
reaction's (EC number, organism) to candidate UniProtKB accessions), plus curated
cytochrome-P450 (CYP) reactions from any organism.

Three provenances, distinguished by the ``accession_source`` column:
- ``curated``  — the accession came from the raw CSV's ``protein_refs`` (a direct
  BRENDA -> UniProt link), exactly as in EnzymeMap_with_seq.
- ``resolved`` — the row had no accession; its (EC, organism) was looked up in
  ``resolved_accessions.csv`` (produced by ``resolve_accessions.py`` via
  ``epp_core.data.search_accessions``) and the best-reviewed candidate attached.
- ``cyp``      — an all-organism CYP reaction from ``raw/cyp_reactions.csv`` (the
  AllOrganism-CYP table copied from the sibling metabolite-prediction repo). The
  enzyme protein sequence is inline (the source ``blast`` column), so these rows need
  no UniProt fetch. ``organism_class`` tags each human/animal/plant/microorganism.

Resolved accessions are candidate enzymes of the right (EC, organism) — correct
activity + species — but are NOT verified to catalyze the specific reaction. Filter
on ``accession_source == "curated"`` for a high-precision cut; use ``resolved`` rows
to expand training coverage. See README.md.

**Human CYP rows are reserved for testing**: every ``accession_source == "cyp"`` row
with ``organism_class == "human"`` is forced into ``split == "test"`` and
``enzyme_split == "test"`` (whole reaction-group / enzyme-cluster moves, so neither
split leaks). Non-human CYP rows flow through the normal grouped split as training
augmentation. The held-out human CYP eval set is exactly
``(accession_source == "cyp") & (organism_class == "human")``.

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
# Curated all-organism cytochrome-P450 reactions (enzyme sequence inline in `blast`),
# copied from the sibling metabolite-prediction repo's
# data/AllOrganism-CYP-v1/raw/reactions.csv. latin-1 / cp1252 encoded (not UTF-8).
CYP_RAW = HERE / "raw" / "cyp_reactions.csv"

DATASET_ID = "enzymemap-with-seq-plus-v1"
SOURCE = "enzymemap-v2-brenda2023"
CYP_SOURCE = "allorganism-cyp"
# Synthetic rxn_idx for CYP rows so they never collide with EnzymeMap's 0..N indices.
CYP_RXN_IDX_OFFSET = 100_000_000

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
    # CYP rows carry an organism class (human/animal/plant/microorganism) and an inline
    # protein sequence; EnzymeMap rows leave these "" (sequence is fetched post-build).
    "organism_class",
    "sequence",
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


def _clean_cyp(value: object) -> str:
    """cyp_reactions.csv cell cleaner: NaN/None and the file's ``/`` placeholder -> ``""``."""
    if value is None or bool(pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text in ("", "/") else text


def _is_human_cyp(txid: object, species1: object) -> bool:
    """True for a CYP row of human origin (Txid 9606 or ``Species1`` contains "Homo sapiens")."""
    if txid is not None and not bool(pd.isna(txid)):
        try:
            if int(txid) == 9606:  # type: ignore[arg-type]
                return True
        except (TypeError, ValueError):
            pass
    return "homo sapiens" in _clean_cyp(species1).lower()


def _organism_class(txid: object, species1: object, species: object) -> str:
    """human / animal / plant / microorganism for a CYP row (``""`` if unknown). Human
    (Txid 9606 / "Homo sapiens") overrides the coarser ``Species`` column."""
    if _is_human_cyp(txid, species1):
        return "human"
    cls = _clean_cyp(species).lower()
    return cls if cls in ("animal", "plant", "microorganism") else ""


def load_cyp_records(cyp_raw: pd.DataFrame) -> list[dict]:
    """CYP path: one record per usable cytochrome-P450 reaction from cyp_reactions.csv.

    Each row's primary ``Substrate1 -> Product1`` transformation is taken (slots 2+ are
    cofactors: O2 / NADPH / H2O ...); the enzyme protein sequence is inline in ``blast``,
    so these rows need no UniProt fetch. ``accession_source="cyp"``; ``organism_class``
    tags human/animal/plant/microorganism (human rows are later forced to the test split).
    Rows missing a substrate, product, sequence, or enzyme key are skipped.
    """
    records: list[dict] = []
    for i, (sub, pro, blast, symbol, uniprot, ec, species1, species, txid) in enumerate(
        zip(
            cyp_raw["sub_Smiles1"],
            cyp_raw["pro_Smiles1"],
            cyp_raw["blast"],
            cyp_raw["Symbol"],
            cyp_raw["Uniprot"],
            cyp_raw["EC number"],
            cyp_raw["Species1"],
            cyp_raw["Species"],
            cyp_raw["Txid"],
            strict=True,
        )
    ):
        reactant, product, sequence = _clean_cyp(sub), _clean_cyp(pro), _clean_cyp(blast)
        if not reactant or not product or not sequence:
            continue  # need a substrate, a product, and an enzyme sequence
        accession = _clean_cyp(uniprot) or _clean_cyp(symbol)
        if not accession:
            continue  # need a stable embedding/dedup key
        records.append(
            {
                "reaction": f"{reactant}>>{product}",
                "rxn_idx": CYP_RXN_IDX_OFFSET + i,
                "ec_num": _clean_cyp(ec),
                "organism": _clean_cyp(species1),
                "organism_class": _organism_class(txid, species1, species),
                "direction": "forward",
                "quality": 1.0,
                "natural": True,
                "steps": "single",
                "uniprot_id": accession,
                "accession_source": "cyp",
                "sequence": sequence,
            }
        )
    return records


def force_groups_to_test(
    df: pd.DataFrame, group_col: str, split_col: str, selector: pd.Series
) -> pd.DataFrame:
    """Move every row whose ``group_col`` value appears in any ``selector`` row to the test
    split. ``selector`` is a boolean mask over ``df``. Whole groups move together, so a group
    never straddles the partition — the existing leakage invariant still holds. Mutates and
    returns ``df``."""
    groups = df.loc[selector, group_col].unique()
    df.loc[df[group_col].isin(groups), split_col] = "test"
    return df


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
    if not CYP_RAW.exists():
        raise FileNotFoundError(
            f"{CYP_RAW} not found. Copy it from the sibling metabolite-prediction repo:\n"
            f"  cp <…>/metabolite-prediction/data/AllOrganism-CYP-v1/raw/reactions.csv "
            f"{CYP_RAW}\nSee data/EnzymeMap_with_seq_plus/README.md."
        )
    PROCESSED.mkdir(exist_ok=True)

    raw = pd.read_csv(RAW)
    n_raw = len(raw)
    resolved_table = pd.read_csv(RESOLVED_TABLE)
    cyp_raw = pd.read_csv(CYP_RAW, encoding="latin-1")  # cp1252, not UTF-8

    # Curated FIRST, then resolved, then CYP: on a (reaction, uniprot_id) dedup collision
    # build_reactions keeps the first-seen, so the EnzymeMap provenance wins (accession_source
    # is not in the dedup key).
    records = load_records(raw)
    records += load_resolved_records(
        raw, resolved_table, explode_all=(MULTI_ACCESSION == "explode_all")
    )
    cyp_records = load_cyp_records(cyp_raw)
    records += cyp_records

    df, stats = build_reactions(
        records,
        max_tokens=MAX_TOKENS,
        source=SOURCE,
        extra_cols=EXTRA_COLS,
        dedup_extra_cols=DEDUP_EXTRA_COLS,
    )
    n_pairs = len(df)
    n_unique_accessions = _to_int(df["uniprot_id"].nunique())

    # CYP rows already carry an inline sequence (from the source `blast` column); only
    # EnzymeMap curated/resolved rows (sequence == "") need a UniProt fetch. Fetch that
    # subset once (the EnzymeMap_with_seq cache makes already-seen ones free), then fill
    # only the empties so inline CYP sequences are left intact.
    need = sorted({u for u, s in zip(df["uniprot_id"], df["sequence"], strict=True) if not s})
    print(f"Fetching sequences for {len(need)} accessions without an inline sequence (cached)...")
    fetched = fetch_sequences(need, cache_path=SEQ_CACHE)
    n_resolved_direct = len(fetched)

    # Recover accessions the live endpoint dropped (obsolete/secondary) from the
    # UniParc archive — one lookup per missing accession (cached, resumable).
    missing = [a for a in need if a not in fetched]
    n_recovered_uniparc = 0
    if missing:
        print(f"Recovering {len(missing)} missing accessions from the UniParc archive...")
        recovered = uniparc_sequences(missing, cache_path=UNIPARC_CACHE)
        n_recovered_uniparc = len(recovered)
        fetched = {**fetched, **recovered}
        print(f"  recovered {n_recovered_uniparc} of {len(missing)} via UniParc")

    filled = df["uniprot_id"].map(lambda a: fetched.get(a, ""))
    df["sequence"] = df["sequence"].where(df["sequence"] != "", filled)
    df = cast(pd.DataFrame, df[df["sequence"] != ""].reset_index(drop=True))
    df["seq_len"] = df["sequence"].str.len()
    n_dropped_no_seq = n_pairs - len(df)
    df["reaction_id"] = range(len(df))  # re-contiguous after the drop
    df.loc[df["accession_source"] == "cyp", "source"] = CYP_SOURCE  # correct provenance tag

    # Group splits on the direction-collapsed reaction identity (no forward/reverse leak).
    df["_undirected"] = df.apply(
        lambda r: undirected_reaction_key(
            r["reactant_smiles"].split("."), r["product_smiles"].split(".")
        ),
        axis=1,
    )
    df = assign_splits(df, group_col="_undirected", fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED)

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

    # Reserve human CYP reactions for testing: force their whole reaction-group AND whole
    # enzyme-cluster to the test split. Whole-group moves keep both splits leakage-free (the
    # asserts below still hold). A few non-human / EnzymeMap rows may be pulled into test by
    # sharing a group with a human CYP row (counted in the manifest).
    human_cyp = (df["accession_source"] == "cyp") & (df["organism_class"] == "human")
    n_human_forced_test = _to_int(human_cyp.sum())
    split_before = df["split"].copy()
    force_groups_to_test(df, "_undirected", "split", human_cyp)
    force_groups_to_test(df, "enzyme_cluster", "enzyme_split", human_cyp)
    n_pulled_into_test = _to_int(
        (((df["split"] == "test") & (split_before != "test")) & ~human_cyp).sum()
    )

    spans = df.groupby("_undirected")["split"].nunique()
    if _to_int((spans > 1).sum()):
        raise SystemExit("LEAKAGE DETECTED: a reaction identity spans multiple splits")
    cluster_spans = df.groupby("enzyme_cluster")["enzyme_split"].nunique()
    if _to_int((cluster_spans > 1).sum()):
        raise SystemExit("LEAKAGE DETECTED: an enzyme cluster spans multiple splits")
    if n_human_forced_test and not (
        (df.loc[human_cyp, "split"] == "test").all()
        and (df.loc[human_cyp, "enzyme_split"] == "test").all()
    ):
        raise SystemExit("human CYP rows are not fully held out in the test split")
    df = df.drop(columns=["_undirected"])

    df.to_parquet(PROCESSED / "reactions.parquet", index=False)

    per_split = {s: _to_int((df["split"] == s).sum()) for s in VALID_SPLITS}
    n_curated = _to_int((df["accession_source"] == "curated").sum())
    n_resolved = _to_int((df["accession_source"] == "resolved").sum())
    is_cyp = df["accession_source"] == "cyp"
    n_cyp = _to_int(is_cyp.sum())
    n_cyp_human = _to_int((is_cyp & (df["organism_class"] == "human")).sum())
    n_cyp_non_human = n_cyp - n_cyp_human
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
        "accession_source_counts": {"curated": n_curated, "resolved": n_resolved, "cyp": n_cyp},
        "cyp": {
            "raw_file": CYP_RAW.name,
            "raw_sha256": sha256_file(CYP_RAW),
            "source": CYP_SOURCE,
            "n_cyp_rows_loaded": len(cyp_records),
            "n_cyp_kept": n_cyp,
            "n_cyp_non_human": n_cyp_non_human,
            "n_cyp_human": n_cyp_human,
            "n_human_forced_test": n_human_forced_test,
            "n_rows_pulled_into_test_by_human_collision": n_pulled_into_test,
        },
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
        "n_unique_accessions": n_unique_accessions,
        "n_accessions_need_fetch": len(need),
        "n_resolved_direct": n_resolved_direct,
        "n_recovered_uniparc": n_recovered_uniparc,
        "n_accessions_resolved": len(fetched),
        "n_dropped_no_seq": n_dropped_no_seq,
        "n_out": len(df),
        "per_split": per_split,
        "n_ec_numbers": _to_int(df["ec_num"].nunique()),
    }
    write_json(PROCESSED / "build_manifest.json", manifest)

    print(f"Built {len(df)} (reaction, protein) examples -> {PROCESSED}")
    print(
        f"  provenance: {n_curated} curated, {n_resolved} resolved, {n_cyp} cyp "
        f"({n_cyp_non_human} non-human, {n_cyp_human} human held out for test)"
    )
    print(f"  per split: {per_split}")
    print(
        f"  fetched {len(fetched)}/{len(need)} non-inline accessions, "
        f"{n_dropped_no_seq} pairs dropped (no sequence)"
    )


if __name__ == "__main__":
    main()
