import torch
from esm2_150m_reactiont5.data import ProteinSeq2SeqCollator, format_input


def test_format_input_wraps_reactants():
    assert format_input("CCO.CC=O") == "REACTANT:CCO.CC=OREAGENT:"


class _FakeTokenizer:
    """Minimal stand-in for ``tokenizer.pad`` (no model download)."""

    def pad(self, features, return_tensors=None):
        max_len = max(len(f["input_ids"]) for f in features)
        ids = torch.zeros((len(features), max_len), dtype=torch.long)
        mask = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            ids[i, :n] = torch.as_tensor(f["input_ids"])
            mask[i, :n] = torch.as_tensor(f["attention_mask"])
        return {"input_ids": ids, "attention_mask": mask}


def test_collator_pads_both_token_streams_independently():
    # SMILES stream lengths {3, 2}; ESM stream lengths {5, 4} -> padded separately.
    features = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [4, 5],
            "esm_input_ids": [0, 10, 11, 12, 2],
            "esm_attention_mask": [1, 1, 1, 1, 1],
        },
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [4],
            "esm_input_ids": [0, 10, 11, 2],
            "esm_attention_mask": [1, 1, 1, 1],
        },
    ]
    batch = ProteinSeq2SeqCollator(_FakeTokenizer(), _FakeTokenizer())(features)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 2)
    assert batch["labels"][1, 1].item() == -100  # shorter label padded with -100

    # The ESM stream is padded to its own max length (5), not the SMILES one (3).
    assert batch["esm_input_ids"].shape == (2, 5)
    assert batch["esm_attention_mask"].shape == (2, 5)
    assert batch["esm_attention_mask"][1].tolist() == [1, 1, 1, 1, 0]  # 2nd seq padded
