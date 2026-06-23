"""Fast pure-logic tests for the GUI backend (app/). No model loading."""

from __future__ import annotations

import pandas as pd
import pytest
from app.registry import DATASETS, GENERATIVE, MODELS

from app import registry, render

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_mol_to_png_valid_and_invalid() -> None:
    png = render.mol_to_png("CCO")
    assert isinstance(png, bytes) and png[:8] == _PNG_MAGIC
    # A dot-joined multi-molecule side still renders (one image, two fragments).
    assert isinstance(render.mol_to_png("CCO.O=O"), bytes)
    assert render.mol_to_png("not_a_smiles") is None
    assert render.mol_to_png("") is None


def test_diff_to_png_handles_valid_and_invalid() -> None:
    assert isinstance(render.diff_to_png("CCO", "CCOC"), bytes)  # highlights the change
    assert render.diff_to_png("CCO", "###") is None  # unparseable target
    assert isinstance(render.diff_to_png("###", "CCO"), bytes)  # bad ref → plain render


def test_models_reference_known_datasets() -> None:
    for spec in MODELS.values():
        assert spec.dataset in DATASETS
        assert spec.kind == GENERATIVE


def _reactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reactant_smiles": ["CCO.O=O", "c1ccccc1"],
            "product_smiles": ["CC=O", "Oc1ccccc1"],
            "ec_num": ["1.1.1.1", "1.14.13.1"],
            "organism": ["Homo sapiens", "Escherichia coli"],
            "uniprot_id": ["P00325", "P0ABC1"],
            "sequence": ["MKT", "MAA"],
            "seq_len": [3, 3],
            "direction": ["forward", "forward"],
            "reaction_id": [1, 2],
        }
    )


def test_build_entries_carries_enzyme_and_splits_sides() -> None:
    entries = registry.build_entries(_reactions_df())
    assert len(entries) == 2
    first = entries[0]
    assert first.reactants == ["CCO", "O=O"]  # dot-joined substrate side split out
    assert first.references == ["CC=O"]
    assert first.ec_num == "1.1.1.1"
    assert first.organism == "Homo sapiens"
    assert first.uniprot_id == "P00325"
    assert "EC 1.1.1.1" in first.label and "P00325" in first.label


def test_build_entries_requires_reactants() -> None:
    with pytest.raises(ValueError, match="reactant_smiles"):
        registry.build_entries(pd.DataFrame({"product_smiles": ["CCO"]}))


def test_build_entries_tolerates_missing_enzyme_columns() -> None:
    entries = registry.build_entries(pd.DataFrame({"reactant_smiles": ["CCO"]}))
    assert entries[0].references == [] and entries[0].ec_num == ""


def test_match_products_exact_set_and_tanimoto() -> None:
    # Same molecule, different spelling → exact (canonical) set match.
    exact = registry.match_products("OCC", ["CCO"])
    assert exact.is_exact and exact.best_tanimoto == pytest.approx(1.0)

    # Order-independent multi-molecule set match.
    assert registry.match_products("CC=O.CCO", ["CCO", "CC=O"]).is_exact

    miss = registry.match_products("c1ccccc1", ["CCO"])
    assert not miss.is_exact
    assert miss.best_tanimoto is not None and 0.0 <= miss.best_tanimoto <= 1.0

    assert registry.match_products("CCO", []).best_tanimoto is None


def test_recovered_products_counts_canonical_overlap() -> None:
    recovered, all_true = registry.recovered_products(["CC=O", "CCC"], ["CC=O", "c1ccccc1"])
    assert recovered == {"CC=O"} and len(all_true) == 2


def test_enzyme_options_dedupe_by_uniprot() -> None:
    df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P1", "P2"],
            "ec_num": ["1.1.1.1", "1.1.1.1", "2.2.2.2"],
            "organism": ["a", "a", "b"],
            "sequence": ["MKT", "MKT", "MAA"],
            "seq_len": [3, 3, 3],
        }
    )
    opts = registry.enzyme_options(df)
    assert [o.uniprot_id for o in opts] == ["P1", "P2"]
    assert "EC 2.2.2.2" in opts[1].label
