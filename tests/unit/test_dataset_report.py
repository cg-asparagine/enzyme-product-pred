"""Tests for the dataset/split report and the split_col plumbing (synthetic data)."""

from __future__ import annotations

import pandas as pd
import pytest

from epp_core.data import assign_splits, load_reactions
from epp_core.io import write_json
from epp_core.report.dataset import _ec_class, _overlap, build_dataset_report


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for i in range(12):
        split = ["train"] * 8 + ["valid"] * 2 + ["test"] * 2
        rows.append(
            {
                "reaction_id": i,
                "reactant_smiles": "CCO",
                "product_smiles": "CC=O",
                "n_reactants": 1 + i % 2,
                "n_products": 1,
                "src_n_tokens": 10 + i,
                "tgt_n_tokens": 8 + i,
                "rxn_idx": i % 6,
                "ec_num": ["1.1.1.1", "2.7.1.1", "3.1.1.1"][i % 3],
                "organism": ["E. coli", "yeast"][i % 2],
                "uniprot_id": f"P{i:05d}",
                "sequence": "ACDEFGHIKL" * (3 + i % 4),
                "seq_len": 10 * (3 + i % 4),
                "split": split[i],
                "enzyme_cluster": i // 2,
                "enzyme_split": split[i],
            }
        )
    return pd.DataFrame(rows)


def test_ec_class() -> None:
    assert _ec_class("1.1.1.1") == "1"
    assert _ec_class("7.6.2.1") == "7"
    assert _ec_class("") == "other"
    assert _ec_class("nan") == "other"


def test_overlap_bounds_and_value() -> None:
    df = pd.DataFrame(
        {
            "split": ["train", "train", "test", "test"],
            "uniprot_id": ["A", "B", "A", "C"],  # 1 of 2 test enzymes (A) seen in train
        }
    )
    assert _overlap(df, "split", "uniprot_id") == pytest.approx(0.5)


def test_assign_splits_custom_column_no_straddle() -> None:
    df = pd.DataFrame({"g": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]})
    out = assign_splits(df, group_col="g", split_col="enzyme_split")
    assert "enzyme_split" in out.columns and "split" not in out.columns
    # whole groups stay together
    assert (out.groupby("g")["enzyme_split"].nunique() == 1).all()


def test_load_reactions_split_col(tmp_path) -> None:
    _synthetic_frame().to_parquet(tmp_path / "reactions.parquet", index=False)
    train = load_reactions(tmp_path, "train", split_col="enzyme_split")
    assert len(train) == 8
    assert set(train["enzyme_split"]) == {"train"}


def test_build_dataset_report_writes_pdf(tmp_path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    _synthetic_frame().to_parquet(proc / "reactions.parquet", index=False)
    write_json(
        proc / "build_manifest.json",
        {"dataset_id": "synthetic", "enzyme_split": {"method": "kmer", "k": 4, "threshold": 0.4}},
    )
    out = build_dataset_report(proc)
    assert out.exists() and out.stat().st_size > 1000
    assert (proc / "report_assets" / "enzyme_leakage.png").exists()
