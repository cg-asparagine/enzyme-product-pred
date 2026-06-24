import pandas as pd
from esm2_reactiont5.pipeline import _shuffle_enzyme_conditioning


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reactant_smiles": ["CCO", "CCC", "CCN", "CCBr", "CCF"],
            "product_smiles": ["CC=O", "CCC=O", "CC=N", "CC=Br", "CC=F"],
            "uniprot_id": ["A", "A", "B", "C", "C"],
            "sequence": ["sA", "sA", "sB", "sC", "sC"],
        }
    )


def test_shuffle_enzyme_conditioning_preserves_invariants():
    original = _frame()
    shuffled = _shuffle_enzyme_conditioning(original, seed=0)

    # Reactants/products stay put — only the enzyme conditioning moves.
    assert shuffled["reactant_smiles"].tolist() == original["reactant_smiles"].tolist()
    assert shuffled["product_smiles"].tolist() == original["product_smiles"].tolist()

    # Each id still carries its own sequence (the pair moves together), and the
    # multiset of enzymes is unchanged — it is a permutation, not a remapping.
    seq_by_id = dict(zip(original["uniprot_id"], original["sequence"], strict=True))
    for uid, seq in zip(shuffled["uniprot_id"], shuffled["sequence"], strict=True):
        assert seq_by_id[uid] == seq
    assert sorted(shuffled["uniprot_id"]) == sorted(original["uniprot_id"])

    # Does not mutate the input frame.
    assert original["uniprot_id"].tolist() == ["A", "A", "B", "C", "C"]


def test_shuffle_enzyme_conditioning_is_deterministic():
    frame = _frame()
    a = _shuffle_enzyme_conditioning(frame, seed=42)
    b = _shuffle_enzyme_conditioning(frame, seed=42)
    assert a["uniprot_id"].tolist() == b["uniprot_id"].tolist()
