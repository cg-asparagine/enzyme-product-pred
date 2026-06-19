"""Shared pytest fixtures and configuration."""

import os

# Force a non-interactive matplotlib backend for any test that renders plots.
# Set via env (before matplotlib is imported anywhere) to avoid display errors.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402


@pytest.fixture
def valid_smiles() -> list[str]:
    return [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
        "CCBr",  # bromoethane (multi-char atom Br)
        "ClCCl",  # dichloromethane (multi-char atom Cl)
        "C[C@H](N)C(=O)O",  # L-alanine (stereochemistry)
        "[O-]C(=O)C",  # acetate (charged bracket atom)
        "c1cc[nH]c1",  # pyrrole (bracket aromatic atom)
    ]


@pytest.fixture
def invalid_smiles() -> list[str]:
    return ["", "(((", "[Zz]", "C1CCC"]
