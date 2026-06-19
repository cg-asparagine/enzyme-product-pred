import pandas as pd
import pytest

from epp_core.data import build_reactions, content_hash
from epp_core.data.schema import REACTION_COLUMNS

# "OCC>>CC=O" and "CCO>>CC=O" canonicalize identically -> duplicate reactions.
RECORDS = [
    {"reaction": "OCC>>CC=O", "ec_num": "1.1.1.1"},
    {"reaction": "CCO>>CC=O", "ec_num": "1.1.1.1"},  # duplicate of the above
    {"reaction": "CCC>>(((", "ec_num": "1.1.1.1"},  # invalid product
    {"reaction": "CCO>>CCO", "ec_num": "1.1.1.1"},  # identity reaction
    {"reaction": "no-arrow", "ec_num": "1.1.1.1"},  # malformed (no '>>')
    {"reaction": "c1ccccc1>>Oc1ccccc1", "ec_num": "1.14.13.1"},
]


def test_build_dedupes_and_filters():
    df, stats = build_reactions(RECORDS, source="test", extra_cols=["ec_num"])
    assert stats.n_in == 6
    assert stats.n_invalid == 2  # invalid product + malformed
    assert stats.n_identity == 1
    assert stats.n_duplicate == 1
    assert stats.n_out == 2
    assert len(df) == 2
    assert set(REACTION_COLUMNS).issubset(df.columns)
    assert "ec_num" in df.columns
    assert "CCO" in set(df["reactant_smiles"])  # canonicalized (OCC -> CCO)


def test_build_is_deterministic():
    df1, _ = build_reactions(RECORDS, extra_cols=["ec_num"])
    df2, _ = build_reactions(RECORDS, extra_cols=["ec_num"])
    pd.testing.assert_frame_equal(df1, df2)
    assert content_hash(df1) == content_hash(df2)


def test_content_hash_changes_with_data():
    df1, _ = build_reactions(RECORDS, extra_cols=["ec_num"])
    df2, _ = build_reactions(
        RECORDS + [{"reaction": "CCCC>>CCCCO", "ec_num": "1.1.1.1"}], extra_cols=["ec_num"]
    )
    assert content_hash(df1) != content_hash(df2)


def test_max_tokens_filter():
    long_rxn = [{"reaction": "C" * 50 + ">>CCO"}]
    _, stats = build_reactions(long_rxn, max_tokens=10)
    assert stats.n_too_long == 1
    assert stats.n_out == 0


def test_identity_kept_when_drop_identity_false():
    _, stats = build_reactions([{"reaction": "CCO>>CCO"}], drop_identity=False)
    assert stats.n_identity == 0
    assert stats.n_out == 1


def test_dedup_extra_cols_keeps_per_ec_rows():
    # Same reaction catalyzed by two EC numbers: collapsed by default, kept when
    # the EC is part of the dedup key (the per-enzyme training signal survives).
    records = [
        {"reaction": "CCO>>CC=O", "ec_num": "1.1.1.1"},
        {"reaction": "CCO>>CC=O", "ec_num": "1.1.1.2"},
    ]
    df_default, _ = build_reactions(records, extra_cols=["ec_num"])
    assert len(df_default) == 1

    df_ec, stats = build_reactions(records, extra_cols=["ec_num"], dedup_extra_cols=["ec_num"])
    assert len(df_ec) == 2
    assert set(df_ec["ec_num"]) == {"1.1.1.1", "1.1.1.2"}
    assert stats.n_duplicate == 0


def test_extra_cols_reject_core_overlap():
    with pytest.raises(ValueError, match="overlap"):
        build_reactions(RECORDS, extra_cols=["reactant_smiles"])


def test_dedup_extra_cols_must_subset_extra_cols():
    with pytest.raises(ValueError, match="subset"):
        build_reactions(RECORDS, dedup_extra_cols=["ec_num"])  # ec_num not in extra_cols
