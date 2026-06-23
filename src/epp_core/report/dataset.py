"""Dataset / split EDA report -> PDF.

Reads a processed reactions dataset (``reactions.parquet`` + ``build_manifest.json``)
and renders a PDF describing the enzyme clusters, the train/valid/test splits, and
dataset-level distributions (sequence length, reaction size, enzyme families). The
report *reads* the baked-in ``enzyme_cluster`` / ``enzyme_split`` columns rather than
recomputing them, so it always describes the split that prepare.py actually wrote.

CLI::

    python -m epp_core.report.dataset --dataset data/<Name>/processed [--out report.pdf]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

from epp_core.chem.reactions import undirected_reaction_key
from epp_core.io import read_json
from epp_core.plots import dataset as dplots
from epp_core.plots import save_figure
from epp_core.report.components import grid_table, image_flowable, kv_table, render_pdf

EC_CLASS_NAMES = {
    "1": "1 Oxidoreductases",
    "2": "2 Transferases",
    "3": "3 Hydrolases",
    "4": "4 Lyases",
    "5": "5 Isomerases",
    "6": "6 Ligases",
    "7": "7 Translocases",
}
_SPLIT_LABELS = {"split": "reaction split", "enzyme_split": "enzyme split"}


def _ec_class(ec: object) -> str:
    head = str(ec).split(".")[0].strip()
    return head if head in EC_CLASS_NAMES else "other"


def _overlap(df: pd.DataFrame, split_col: str, key_col: str) -> float:
    """Fraction of test-split rows whose ``key_col`` value also appears in train."""
    train = set(df.loc[df[split_col] == "train", key_col])
    test = df.loc[df[split_col] == "test", key_col]
    return float(test.isin(train).mean()) if len(test) else 0.0


def _int(x: Any) -> int:
    """Coerce a pandas/numpy scalar reduction to a plain int (stubs widen these to Series)."""
    return int(x)


def _fmt(n: Any) -> str:
    n = float(n)
    return f"{n:,.0f}" if n.is_integer() else f"{n:,.2f}"


def _split_summary_rows(
    df: pd.DataFrame, split_cols: list[str], react_key_col: str | None
) -> list[list[str]]:
    """Per-split-strategy table: examples + enzymes per split, and leakage into test."""
    header = ["split strategy", "train", "valid", "test", "enzyme leak", "reaction leak"]
    rows: list[list[str]] = [header]
    for col in split_cols:
        counts = df[col].value_counts()
        rows.append(
            [
                _SPLIT_LABELS.get(col, col),
                f"{_int(counts.get('train', 0)):,}",
                f"{_int(counts.get('valid', 0)):,}",
                f"{_int(counts.get('test', 0)):,}",
                f"{_overlap(df, col, 'uniprot_id'):.1%}",
                f"{_overlap(df, col, react_key_col):.1%}" if react_key_col else "—",
            ]
        )
    return rows


def _largest_cluster_rows(df: pd.DataFrame, n: int = 12) -> list[list[str]]:
    per_enzyme = df.drop_duplicates("uniprot_id")
    sizes = per_enzyme["enzyme_cluster"].value_counts().head(n)
    rows: list[list[str]] = [["cluster", "enzymes", "examples", "top EC class", "split"]]
    for cid, n_enz in sizes.items():
        members = cast(pd.DataFrame, df[df["enzyme_cluster"] == cid])
        ec_classes = [EC_CLASS_NAMES.get(_ec_class(e), "other") for e in members["ec_num"]]
        top_ec = Counter(ec_classes).most_common(1)[0][0] if ec_classes else "—"
        rows.append(
            [
                str(cid),
                str(_int(n_enz)),
                f"{len(members):,}",
                top_ec,
                str(members["enzyme_split"].iloc[0]),
            ]
        )
    return rows


def build_dataset_report(processed_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Render the dataset/split report PDF and return its path."""
    processed_dir = Path(processed_dir)
    df = pd.read_parquet(processed_dir / "reactions.parquet")
    manifest_path = processed_dir / "build_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    out_path = Path(out_path) if out_path else processed_dir / "dataset_report.pdf"
    assets = out_path.parent / "report_assets"
    styles = getSampleStyleSheet()

    has_clusters = "enzyme_cluster" in df.columns
    has_enzyme_split = "enzyme_split" in df.columns
    split_cols = ["split"] + (["enzyme_split"] if has_enzyme_split else [])

    # Measure reaction leakage on the *same* direction-collapsed identity the
    # reaction split groups on (so the reaction split shows ~0 by construction);
    # fall back to rxn_idx only if SMILES are unavailable.
    react_key_col: str | None = None
    if {"reactant_smiles", "product_smiles"}.issubset(df.columns):
        df = df.copy()
        df["_reaction_key"] = df.apply(
            lambda r: undirected_reaction_key(
                str(r["reactant_smiles"]).split("."), str(r["product_smiles"]).split(".")
            ),
            axis=1,
        )
        react_key_col = "_reaction_key"
    elif "rxn_idx" in df.columns:
        react_key_col = "rxn_idx"

    def para(text: str) -> Paragraph:
        return Paragraph(text, styles["Normal"])

    def fig(figure: Any, name: str, width: float = 5.2) -> Any:
        return image_flowable(save_figure(figure, assets / f"{name}.png"), max_width_in=width)

    per_enzyme = df.drop_duplicates("uniprot_id")
    sections: list[tuple[str, list[Any]]] = []

    # --- Overview ----------------------------------------------------------
    overview = {
        "dataset_id": manifest.get("dataset_id", processed_dir.parent.name),
        "source": manifest.get("source", "—"),
        "examples (reaction, enzyme pairs)": f"{len(df):,}",
        "unique enzymes (UniProt)": f"{df['uniprot_id'].nunique():,}",
        "unique reactions (undirected)": (
            f"{df['_reaction_key'].nunique():,}"
            if react_key_col == "_reaction_key"
            else (f"{df['rxn_idx'].nunique():,}" if "rxn_idx" in df else "—")
        ),
        "unique EC numbers": f"{df['ec_num'].nunique():,}" if "ec_num" in df else "—",
        "unique organisms": f"{df['organism'].nunique():,}" if "organism" in df else "—",
        "enzyme clusters": f"{df['enzyme_cluster'].nunique():,}" if has_clusters else "—",
    }
    sections.append(("Overview", [kv_table(overview)]))

    # --- Splits (the headline) --------------------------------------------
    split_flowables: list[Any] = [
        para(
            "Two independent partitions are stored. <b>split</b> is the v1 "
            "reaction-grouped split (whole undirected reactions kept together). "
            "<b>enzyme_split</b> assigns whole <i>enzyme clusters</i> to a set, so "
            "test enzymes (and their close homologs) are unseen in training — the "
            "honest generalization test for a sequence-conditioned model."
        ),
        grid_table(_split_summary_rows(df, split_cols, react_key_col)),
        para(
            "'enzyme leak' = share of test examples whose enzyme also appears in "
            "train; 'reaction leak' = same for the direction-collapsed reaction "
            "identity. Each split drives its own leak to ~0 by construction: the "
            "reaction split leaves most enzymes shared (high enzyme leak), while the "
            "enzyme split leaves most reactions shared (high reaction leak) — the "
            "deliberate trade-off for testing new-enzyme generalization."
        ),
        fig(
            dplots.plot_split_composition(
                {_SPLIT_LABELS.get(c, c): df[c].value_counts().to_dict() for c in split_cols}
            ),
            "split_composition",
        ),
    ]
    if has_enzyme_split:
        leakage = {_SPLIT_LABELS.get(c, c): _overlap(df, c, "uniprot_id") for c in split_cols}
        split_flowables.append(fig(dplots.plot_enzyme_leakage(leakage), "enzyme_leakage"))
    sections.append(("Train / valid / test splits", split_flowables))

    # --- Enzyme clusters ---------------------------------------------------
    if has_clusters:
        sizes = per_enzyme["enzyme_cluster"].value_counts()
        clu = manifest.get("enzyme_split", {})
        cluster_stats = {
            "clustering method": clu.get("method", "—"),
            "k-mer size (k)": clu.get("k", "—"),
            "Jaccard threshold": clu.get("threshold", "—"),
            "clusters": f"{len(sizes):,}",
            "singleton clusters": f"{_int((sizes == 1).sum()):,} ({(sizes == 1).mean():.1%})",
            "largest cluster (enzymes)": f"{_int(sizes.max()):,}",
            "mean / median size": f"{sizes.mean():.2f} / {_fmt(sizes.median())}",
        }
        sections.append(
            (
                "Enzyme clusters",
                [
                    para(
                        "Enzymes are clustered by k-mer Jaccard similarity "
                        "(greedy, representative-based); whole clusters define the "
                        "enzyme split. Most enzymes are singletons; clustering merges "
                        "the homologous minority so near-twins cannot straddle splits."
                    ),
                    kv_table(cluster_stats),
                    fig(dplots.plot_cluster_size_hist(sizes.tolist()), "cluster_sizes"),
                    para("<b>Largest enzyme clusters</b>"),
                    grid_table(_largest_cluster_rows(df)),
                ],
            )
        )

    # --- Sequences ---------------------------------------------------------
    seq_lens = per_enzyme["seq_len"] if "seq_len" in df else per_enzyme["sequence"].str.len()
    seq_stats = {
        "enzymes": f"{len(per_enzyme):,}",
        "length min / median / max": (
            f"{_int(seq_lens.min()):,} / {_int(seq_lens.median()):,} / {_int(seq_lens.max()):,}"
        ),
        "length mean": f"{seq_lens.mean():.0f}",
    }
    sections.append(
        (
            "Enzyme sequences",
            [
                kv_table(seq_stats),
                fig(dplots.plot_sequence_length_hist(seq_lens.tolist()), "seq_len"),
            ],
        )
    )

    # --- Chemical transformations -----------------------------------------
    chem_flowables: list[Any] = []
    if {"src_n_tokens", "tgt_n_tokens"}.issubset(df.columns):
        chem_flowables.append(
            fig(
                dplots.plot_reaction_token_lengths(
                    df["src_n_tokens"].tolist(), df["tgt_n_tokens"].tolist()
                ),
                "token_lengths",
            )
        )
    if {"n_reactants", "n_products"}.issubset(df.columns):
        chem_stats = {
            "mean reactants / products": (
                f"{df['n_reactants'].mean():.2f} / {df['n_products'].mean():.2f}"
            ),
            "median src / tgt tokens": (
                f"{_fmt(df['src_n_tokens'].median())} / {_fmt(df['tgt_n_tokens'].median())}"
                if "src_n_tokens" in df
                else "—"
            ),
        }
        chem_flowables.insert(0, kv_table(chem_stats))
        chem_flowables.append(
            fig(
                dplots.plot_molecule_counts(
                    {
                        "reactants": df["n_reactants"].value_counts().to_dict(),
                        "products": df["n_products"].value_counts().to_dict(),
                    }
                ),
                "molecule_counts",
            )
        )
    if chem_flowables:
        sections.append(("Chemical transformations", chem_flowables))

    # --- Enzyme families ---------------------------------------------------
    if "ec_num" in df.columns:
        ec_counts_raw = df["ec_num"].map(_ec_class).value_counts()
        ordered = [c for c in (*EC_CLASS_NAMES, "other") if c in ec_counts_raw.index]
        ec_named = {EC_CLASS_NAMES.get(c, "other"): _int(ec_counts_raw[c]) for c in ordered}
        sections.append(
            (
                "Enzyme families",
                [
                    fig(dplots.plot_ec_families(ec_named), "ec_families", width=5.8),
                    kv_table({k: f"{v:,}" for k, v in ec_named.items()}),
                ],
            )
        )

    subtitle = [
        f"Dataset: {overview['dataset_id']}  •  {len(df):,} examples  •  "
        f"{df['uniprot_id'].nunique():,} enzymes",
        f"Source parquet: {processed_dir / 'reactions.parquet'}",
    ]
    return render_pdf(out_path, "Dataset & split report", subtitle, sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a dataset / split EDA report.")
    parser.add_argument("--dataset", required=True, help="path to a dataset's processed/ dir")
    parser.add_argument(
        "--out", default=None, help="output PDF path (default: <dataset>/dataset_report.pdf)"
    )
    args = parser.parse_args()
    out = build_dataset_report(args.dataset, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
