"""Dataset + collator for sequence-conditioned reaction-product prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

REACTANT_PREFIX = "REACTANT:"
REAGENT_PREFIX = "REAGENT:"


def format_input(reactant_smiles: str) -> str:
    """ReactionT5 forward-prediction input: reactants then an empty reagent block."""
    return f"{REACTANT_PREFIX}{reactant_smiles}{REAGENT_PREFIX}"


class ReactionSequenceDataset(Dataset):
    """Tokenizes reactant->product and attaches the enzyme's precomputed embedding."""

    def __init__(
        self,
        frame: Any,
        embeddings: dict[str, np.ndarray],
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.reactants = frame["reactant_smiles"].tolist()
        self.products = frame["product_smiles"].tolist()
        self.uniprot_ids = frame["uniprot_id"].tolist()
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

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
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"],
            "protein_embedding": self.embeddings[self.uniprot_ids[index]],
        }


class ProteinSeq2SeqCollator:
    """Pads the SMILES token fields (labels padded with -100) and stacks embeddings."""

    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        embeddings = torch.as_tensor(
            np.stack([f["protein_embedding"] for f in features]), dtype=torch.float32
        )
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
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "protein_embedding": embeddings,
        }
