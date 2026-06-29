"""Dataset + collator for sequence-conditioned reaction-product prediction.

Differs from the frozen model only in the enzyme channel: instead of attaching a
precomputed embedding vector, each example carries the enzyme sequence tokenized for
ESM, and the collator pads that second token stream alongside the SMILES one. ESM
then embeds it live inside the model's forward.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

REACTANT_PREFIX = "REACTANT:"
REAGENT_PREFIX = "REAGENT:"


def format_input(reactant_smiles: str) -> str:
    """ReactionT5 forward-prediction input: reactants then an empty reagent block."""
    return f"{REACTANT_PREFIX}{reactant_smiles}{REAGENT_PREFIX}"


class ReactionSequenceDataset(Dataset):
    """Tokenizes reactant->product (T5) and the enzyme sequence (ESM) per example."""

    def __init__(
        self,
        frame: Any,
        tokenizer: Any,
        esm_tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
        max_residues: int,
    ) -> None:
        self.reactants = frame["reactant_smiles"].tolist()
        self.products = frame["product_smiles"].tolist()
        self.sequences = frame["sequence"].tolist()
        self.tokenizer = tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_residues = max_residues

    def __len__(self) -> int:
        return len(self.reactants)

    def __getitem__(self, index: int) -> dict[str, Any]:
        model_inputs = self.tokenizer(
            format_input(self.reactants[index]),
            max_length=self.max_input_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=self.products[index],
            max_length=self.max_target_length,
            truncation=True,
        )
        esm = self.esm_tokenizer(
            str(self.sequences[index])[: self.max_residues],
            max_length=self.max_residues + 2,  # + <cls>/<eos>
            truncation=True,
        )
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"],
            "esm_input_ids": esm["input_ids"],
            "esm_attention_mask": esm["attention_mask"],
        }


class ProteinSeq2SeqCollator:
    """Pads the SMILES stream (labels with -100) and the ESM stream independently."""

    def __init__(self, tokenizer: Any, esm_tokenizer: Any, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        label_seqs = [f["labels"] for f in features]
        max_len = max(len(seq) for seq in label_seqs)
        labels = torch.full((len(features), max_len), self.label_pad_token_id, dtype=torch.long)
        for i, seq in enumerate(label_seqs):
            labels[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)

        batch = self.tokenizer.pad(
            [
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                for f in features
            ],
            return_tensors="pt",
        )
        esm_batch = self.esm_tokenizer.pad(
            [
                {"input_ids": f["esm_input_ids"], "attention_mask": f["esm_attention_mask"]}
                for f in features
            ],
            return_tensors="pt",
        )
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "esm_input_ids": esm_batch["input_ids"],
            "esm_attention_mask": esm_batch["attention_mask"],
        }
