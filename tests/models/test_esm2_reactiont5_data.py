import numpy as np
import torch
from esm2_reactiont5.data import ProteinSeq2SeqCollator, format_input


def test_format_input_wraps_reactants():
    assert format_input("CCO.CC=O") == "REACTANT:CCO.CC=OREAGENT:"


class _FakeTokenizer:
    """Minimal stand-in for the collator's tokenizer.pad (no model download)."""

    def pad(self, features, return_tensors=None):
        max_len = max(len(f["input_ids"]) for f in features)
        ids = torch.zeros((len(features), max_len), dtype=torch.long)
        mask = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            ids[i, :n] = torch.as_tensor(f["input_ids"])
            mask[i, :n] = torch.as_tensor(f["attention_mask"])
        return {"input_ids": ids, "attention_mask": mask}


def test_collator_pads_labels_and_stacks_embeddings():
    features = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [4, 5],
            "protein_embedding": np.ones(4, np.float32),
        },
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [4],
            "protein_embedding": np.zeros(4, np.float32),
        },
    ]
    batch = ProteinSeq2SeqCollator(_FakeTokenizer())(features)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 2)
    assert batch["labels"][1, 1].item() == -100  # shorter label padded with -100
    assert batch["protein_embedding"].shape == (2, 4)
    assert batch["protein_embedding"].dtype == torch.float32
