from types import SimpleNamespace

import pandas as pd
import torch.nn as nn
from esm2_150m_reactiont5.pipeline import _esm_param_groups, _shuffle_enzyme_conditioning


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


class _Tiny(nn.Module):
    """esm.* params (incl. a LayerNorm) + task-side params, to exercise grouping."""

    def __init__(self) -> None:
        super().__init__()
        self.esm = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.t5 = nn.Linear(4, 4)
        self.protein_proj = nn.Linear(4, 4)


def test_esm_param_groups_split_lr_and_weight_decay():
    model = _Tiny()
    args = SimpleNamespace(weight_decay=0.01)
    groups = _esm_param_groups(model, args, esm_lr=2e-5, base_lr=1e-4)

    id_to_name = {id(p): n for n, p in model.named_parameters()}
    # No weight decay on LayerNorm weights or any bias (mirrors the stock Trainer).
    expected_no_decay = {"esm.0.bias", "esm.1.weight", "esm.1.bias", "t5.bias", "protein_proj.bias"}

    placed = []
    for g in groups:
        for p in g["params"]:
            name = id_to_name[id(p)]
            placed.append(name)
            # ESM gets the small LR; everything else gets the base LR.
            assert (g["lr"] == 2e-5) == name.startswith("esm.")
            assert g["weight_decay"] == (0.0 if name in expected_no_decay else 0.01)

    # Every trainable parameter is placed in exactly one group.
    assert sorted(placed) == sorted(n for n, _ in model.named_parameters())
    assert len(placed) == len(set(placed))
