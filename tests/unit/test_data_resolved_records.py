"""Tests for EnzymeMap_with_seq_plus's resolved-record join-back + provenance.

The builder lives in a data/ script (not an installed package), so we load it by
path. `load_resolved_records` takes in-memory dataframes, so these are pure-logic
unit tests with no network or file IO.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from epp_core.data import build_reactions

_PREPARE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "EnzymeMap_with_seq_plus" / "prepare.py"
)


def _load_prepare() -> Any:
    spec = importlib.util.spec_from_file_location("ewsp_prepare", _PREPARE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_prepare()


def _raw_df() -> pd.DataFrame:
    # Columns load_resolved_records / load_records read from the raw CSV.
    rows = [
        # A: accession-less (protein_db NaN), resolvable, forward
        dict(
            unmapped="CCO>>CC=O",
            rxn_idx=10,
            ec_num="1.1.1.1",
            organism="Escherichia coli",
            source="enzymemap-v2-brenda2023",
            quality=1.0,
            natural=True,
            steps="single",
            protein_db=float("nan"),
            protein_refs="[]",
        ),
        # B: curated (has accession), same (EC, organism) as A -> excluded from resolved
        dict(
            unmapped="CCCO>>CCC=O",
            rxn_idx=11,
            ec_num="1.1.1.1",
            organism="Escherichia coli",
            source="enzymemap-v2-brenda2023",
            quality=1.0,
            natural=True,
            steps="single",
            protein_db="uniprot",
            protein_refs="['Q00000']",
        ),
        # C: accession-less but empty organism -> excluded
        dict(
            unmapped="CCO>>CC=O",
            rxn_idx=12,
            ec_num="1.1.1.1",
            organism="",
            source="enzymemap-v2-brenda2023",
            quality=1.0,
            natural=True,
            steps="single",
            protein_db=float("nan"),
            protein_refs="[]",
        ),
        # D: accession-less (genbank counts as none), reversed direction, resolvable
        dict(
            unmapped="CCC>>CCCO",
            rxn_idx=13,
            ec_num="2.2.2.2",
            organism="Bacillus subtilis",
            source="enzymemap-v2-brenda2023-reversed",
            quality=0.9,
            natural=False,
            steps="single",
            protein_db="genbank",
            protein_refs="['G1']",
        ),
        # E: accession-less, low quality -> excluded by the quality mask
        dict(
            unmapped="CCCC>>CCCCO",
            rxn_idx=14,
            ec_num="2.2.2.2",
            organism="Bacillus subtilis",
            source="enzymemap-v2-brenda2023",
            quality=0.1,
            natural=True,
            steps="single",
            protein_db=float("nan"),
            protein_refs="[]",
        ),
        # F: accession-less, (EC, organism) not in the resolved table -> skipped
        dict(
            unmapped="CCO>>CC=O",
            rxn_idx=15,
            ec_num="9.9.9.9",
            organism="Nobody",
            source="enzymemap-v2-brenda2023",
            quality=1.0,
            natural=True,
            steps="single",
            protein_db=float("nan"),
            protein_refs="[]",
        ),
    ]
    return pd.DataFrame(rows)


def _resolved_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                ec_num="1.1.1.1", organism="Escherichia coli", n_accessions=3, accessions="P1;P2;P3"
            ),
            dict(ec_num="2.2.2.2", organism="Bacillus subtilis", n_accessions=1, accessions="R1"),
        ]
    )


def test_pick_one_joins_back_one_record_per_reaction():
    recs = prepare.load_resolved_records(_raw_df(), _resolved_df(), explode_all=False)
    by_uid = {r["uniprot_id"]: r for r in recs}
    assert set(by_uid) == {"P1", "R1"}  # A -> first of P1/P2/P3; D -> R1

    a = by_uid["P1"]  # carries the raw row's per-reaction fields
    assert a["rxn_idx"] == 10
    assert a["ec_num"] == "1.1.1.1"
    assert a["organism"] == "Escherichia coli"
    assert a["quality"] == 1.0
    assert a["natural"] is True
    assert a["steps"] == "single"
    assert a["direction"] == "forward"
    assert a["accession_source"] == "resolved"

    assert by_uid["R1"]["direction"] == "reversed"  # direction from the source string


def test_explode_all_emits_one_record_per_candidate():
    recs = prepare.load_resolved_records(_raw_df(), _resolved_df(), explode_all=True)
    assert sorted(r["uniprot_id"] for r in recs) == ["P1", "P2", "P3", "R1"]  # A x3, D x1


def test_curated_lowqual_emptyorg_and_unresolved_rows_excluded():
    recs = prepare.load_resolved_records(_raw_df(), _resolved_df(), explode_all=True)
    assert all(r["uniprot_id"] != "Q00000" for r in recs)  # curated row B never appears
    assert all(r["accession_source"] == "resolved" for r in recs)
    # Only rows A (x3) and D (x1) survive; B/C/E/F are all filtered out.
    assert len(recs) == 4


def test_has_accession_predicate():
    assert prepare._has_accession("uniprot", "['Q1']") is True
    assert prepare._has_accession("swissprot", "['Q1', 'Q2']") is True
    assert prepare._has_accession("uniprot", "[]") is False  # empty refs
    assert prepare._has_accession("genbank", "['G1']") is False  # non-UniProt db
    assert prepare._has_accession(float("nan"), "[]") is False


def test_direction_from_source():
    assert prepare._direction("enzymemap-v2-brenda2023") == "forward"
    assert prepare._direction("enzymemap-v2-brenda2023-reversed") == "reversed"


def test_curated_wins_collision_in_build_reactions():
    # Same canonical reaction + same accession from both sources -> dedup keeps the
    # first-seen. Feeding curated first guarantees curated provenance survives.
    curated = {"reaction": "CCO>>CC=O", "uniprot_id": "P1", "accession_source": "curated"}
    resolved = {"reaction": "CCO>>CC=O", "uniprot_id": "P1", "accession_source": "resolved"}
    df, stats = build_reactions(
        [curated, resolved],
        extra_cols=["uniprot_id", "accession_source"],
        dedup_extra_cols=["uniprot_id"],
    )
    assert len(df) == 1
    assert df.iloc[0]["accession_source"] == "curated"
    assert stats.n_duplicate == 1
