"""Tests for EnzymeMap_with_seq_plus's CYP loader + the human-held-out forcing helper.

The builder lives in a data/ script (not an installed package), so we load it by path.
`load_cyp_records` and `force_groups_to_test` take in-memory data, so these are pure-logic
unit tests with no network or file IO.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

_PREPARE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "EnzymeMap_with_seq_plus" / "prepare.py"
)


def _load_prepare() -> Any:
    spec = importlib.util.spec_from_file_location("ewsp_prepare_cyp", _PREPARE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_prepare()


def _cyp_df() -> pd.DataFrame:
    # Columns load_cyp_records reads from cyp_reactions.csv. Txid mixes ints + NaN (so the
    # column is float, exactly as the real latin-1 file parses).
    rows = [
        # 0: human via Txid 9606 (Species says "Animal" — human override wins)
        {
            "sub_Smiles1": "CCO",
            "pro_Smiles1": "CC=O",
            "blast": "MHUMAN1",
            "Symbol": "CYP3A4",
            "Uniprot": "P08684",
            "EC number": "1.14.14.1",
            "Species1": "Homo sapiens",
            "Species": "Animal",
            "Txid": 9606,
        },
        # 1: human via Species1 text, Txid missing
        {
            "sub_Smiles1": "CCCO",
            "pro_Smiles1": "CCC=O",
            "blast": "MHUMAN2",
            "Symbol": "CYP2D6",
            "Uniprot": "P10635",
            "EC number": "1.14.14.1",
            "Species1": "Homo sapiens (human)",
            "Species": "",
            "Txid": float("nan"),
        },
        # 2: plant, EC "/" -> empty ec_num
        {
            "sub_Smiles1": "C1CCCCC1",
            "pro_Smiles1": "OC1CCCCC1",
            "blast": "MPLANT",
            "Symbol": "CYP726A20",
            "Uniprot": "A0A067LCX8",
            "EC number": "/",
            "Species1": "Jatropha curcas",
            "Species": "Plant",
            "Txid": 180498,
        },
        # 3: microorganism, Uniprot "/" -> uniprot_id falls back to Symbol
        {
            "sub_Smiles1": "CCCC",
            "pro_Smiles1": "CCCCO",
            "blast": "MMICRO",
            "Symbol": "CYP105",
            "Uniprot": "/",
            "EC number": "1.14.15.1",
            "Species1": "Streptomyces",
            "Species": "Microorganism",
            "Txid": 1883,
        },
        # 4: animal
        {
            "sub_Smiles1": "CCCCC",
            "pro_Smiles1": "CCCCCO",
            "blast": "MANIMAL",
            "Symbol": "CYP6",
            "Uniprot": "Q9XXXX",
            "EC number": "1.14.14.1",
            "Species1": "Mus musculus",
            "Species": "Animal",
            "Txid": 10090,
        },
        # 5: missing substrate -> skipped
        {
            "sub_Smiles1": "/",
            "pro_Smiles1": "CC=O",
            "blast": "MX",
            "Symbol": "CYPX",
            "Uniprot": "PX",
            "EC number": "1.1.1.1",
            "Species1": "Foo",
            "Species": "Plant",
            "Txid": 1,
        },
        # 6: missing sequence (blast "/") -> skipped
        {
            "sub_Smiles1": "CCO",
            "pro_Smiles1": "CC=O",
            "blast": "/",
            "Symbol": "CYPY",
            "Uniprot": "PY",
            "EC number": "1.1.1.1",
            "Species1": "Bar",
            "Species": "Plant",
            "Txid": 2,
        },
    ]
    return pd.DataFrame(rows)


def test_load_cyp_records_basic_fields():
    recs = prepare.load_cyp_records(_cyp_df())
    assert len(recs) == 5  # 7 rows, rows 5 & 6 dropped (missing substrate / sequence)
    by_uid = {r["uniprot_id"]: r for r in recs}
    assert set(by_uid) == {"P08684", "P10635", "A0A067LCX8", "CYP105", "Q9XXXX"}

    r0 = by_uid["P08684"]
    assert r0["reaction"] == "CCO>>CC=O"
    assert r0["sequence"] == "MHUMAN1"  # taken from the `blast` column
    assert r0["accession_source"] == "cyp"
    assert r0["organism_class"] == "human"
    assert r0["organism"] == "Homo sapiens"
    assert r0["direction"] == "forward"
    assert r0["quality"] == 1.0
    assert r0["natural"] is True
    assert r0["steps"] == "single"
    assert r0["ec_num"] == "1.14.14.1"
    assert r0["rxn_idx"] == prepare.CYP_RXN_IDX_OFFSET  # offset + index 0


def test_load_cyp_records_organism_classification():
    by_uid = {r["uniprot_id"]: r for r in prepare.load_cyp_records(_cyp_df())}
    assert by_uid["P10635"]["organism_class"] == "human"  # via "Homo sapiens" in Species1
    assert by_uid["A0A067LCX8"]["organism_class"] == "plant"
    assert by_uid["CYP105"]["organism_class"] == "microorganism"
    assert by_uid["Q9XXXX"]["organism_class"] == "animal"


def test_load_cyp_records_uniprot_fallback_and_empty_ec():
    by_uid = {r["uniprot_id"]: r for r in prepare.load_cyp_records(_cyp_df())}
    # Uniprot "/" -> fall back to the CYP Symbol as the embedding/dedup key
    assert "CYP105" in by_uid
    assert by_uid["CYP105"]["ec_num"] == "1.14.15.1"
    # EC "/" -> empty ec_num (the data report tolerates empty EC)
    assert by_uid["A0A067LCX8"]["ec_num"] == ""


def test_load_cyp_records_skips_incomplete_rows():
    recs = prepare.load_cyp_records(_cyp_df())
    assert all(r["uniprot_id"] not in ("PX", "PY") for r in recs)


def test_force_groups_to_test_moves_whole_group():
    df = pd.DataFrame(
        [
            # group g1: a human CYP row + a non-human CYP twin in the same reaction group
            {"grp": "g1", "split": "train", "accession_source": "cyp", "organism_class": "human"},
            {"grp": "g1", "split": "train", "accession_source": "cyp", "organism_class": "plant"},
            # g2: unrelated rows -> untouched
            {"grp": "g2", "split": "train", "accession_source": "curated", "organism_class": ""},
            # g3: non-human CYP only -> not selected, stays put
            {"grp": "g3", "split": "valid", "accession_source": "cyp", "organism_class": "animal"},
        ]
    )
    selector = (df["accession_source"] == "cyp") & (df["organism_class"] == "human")
    out = prepare.force_groups_to_test(df, "grp", "split", selector)

    # The whole g1 group moves to test — including the non-human twin pulled along.
    assert (out.loc[out["grp"] == "g1", "split"] == "test").all()
    # Groups without a selected (human CYP) row are untouched.
    assert (out.loc[out["grp"] == "g2", "split"] == "train").all()
    assert (out.loc[out["grp"] == "g3", "split"] == "valid").all()
