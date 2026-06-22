"""Train and evaluate ESM2-650M-frozen-ReactionT5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from epp_core.data import content_hash, load_reactions, train_product_smiles
from epp_core.eval.types import GenerativeEvalInputs, TaskType
from epp_core.metadata.capture import ExperimentMetadata
from epp_core.runner import evaluate_model

from .config import TrainConfig
from .data import ProteinSeq2SeqCollator, ReactionSequenceDataset, format_input
from .embeddings import ESM_MODEL_ID, ensure_embeddings, select_device
from .model import ReactionT5WithProtein

_MODEL_NAME = "ESM2-650M-frozen-ReactionT5"


def _subset(frame: Any, n: int | None) -> Any:
    return frame if n is None else frame.head(n)


def _resolve_device(config: TrainConfig) -> str:
    return select_device() if config.device == "auto" else config.device


def _needed_embeddings(
    config: TrainConfig, frames: list[Any], device: str
) -> dict[str, np.ndarray]:
    """Embed exactly the enzyme sequences used by these frames (cache-backed)."""
    id_to_seq: dict[str, str] = {}
    for frame in frames:
        for uid, seq in zip(frame["uniprot_id"], frame["sequence"], strict=True):
            id_to_seq[str(uid)] = str(seq)
    return ensure_embeddings(config.embedding_cache, id_to_seq, device=device)


def train_model(config: TrainConfig) -> str:
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    device = _resolve_device(config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_checkpoint)
    train_df = _subset(load_reactions(config.dataset_dir, "train"), config.max_train_samples)
    valid_df = _subset(load_reactions(config.dataset_dir, "valid"), config.max_eval_samples)
    embeddings = _needed_embeddings(config, [train_df, valid_df], device)

    train_ds = ReactionSequenceDataset(
        train_df, embeddings, tokenizer, config.max_input_length, config.max_target_length
    )
    collator = ProteinSeq2SeqCollator(tokenizer)
    model = ReactionT5WithProtein.from_base(config.base_checkpoint, config.esm_dim)

    args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy="no",
        seed=config.seed,
        fp16=config.fp16,
        remove_unused_columns=False,
        label_names=["labels"],
        use_cpu=(device == "cpu"),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    save_dir = Path(config.output_dir)
    model.save(save_dir)
    tokenizer.save_pretrained(save_dir)
    return str(save_dir)


def _generate(
    model: ReactionT5WithProtein,
    tokenizer: Any,
    frame: Any,
    embeddings: dict[str, np.ndarray],
    config: TrainConfig,
    device: str,
) -> list[list[str]]:
    reactants = frame["reactant_smiles"].tolist()
    uniprot_ids = frame["uniprot_id"].tolist()
    k, batch = config.num_return_sequences, config.per_device_eval_batch_size
    predictions: list[list[str]] = []
    for start in range(0, len(reactants), batch):
        srcs = [format_input(r) for r in reactants[start : start + batch]]
        enc = tokenizer(
            srcs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_input_length,
        ).to(device)
        emb = torch.as_tensor(
            np.stack([embeddings[u] for u in uniprot_ids[start : start + batch]]),
            dtype=torch.float32,
            device=device,
        )
        generated = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            protein_embedding=emb,
            num_beams=config.num_beams,
            num_return_sequences=k,
            max_length=config.max_target_length,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i in range(len(srcs)):
            predictions.append(decoded[i * k : (i + 1) * k])
    return predictions


def evaluate_run(
    config: TrainConfig, model_dir: str | None = None, run_root: str = "experiments"
) -> str:
    from transformers import AutoTokenizer

    resolved = model_dir or config.output_dir
    device = _resolve_device(config)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = ReactionT5WithProtein.from_pretrained_dir(resolved, config.esm_dim)
    model.to(device)
    model.eval()

    test_df = _subset(load_reactions(config.dataset_dir, "test"), config.max_eval_samples)
    embeddings = _needed_embeddings(config, [test_df], device)
    references = [str(p).split(".") for p in test_df["product_smiles"]]
    predictions = _generate(model, tokenizer, test_df, embeddings, config, device)
    inputs = GenerativeEvalInputs(
        references=references,
        predictions=predictions,
        train_smiles=train_product_smiles(config.dataset_dir),
    )

    architecture = {
        "base_checkpoint": config.base_checkpoint,
        "esm_model": ESM_MODEL_ID,
        "esm_frozen": True,
        "esm_dim": config.esm_dim,
        "d_model": int(model.t5.config.d_model),
        "num_trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    metadata = ExperimentMetadata(
        model_name=_MODEL_NAME,
        task_type=TaskType.GENERATIVE,
        hyperparameters=config.as_dict(),
        architecture=architecture,
        train_config={
            "base_checkpoint": config.base_checkpoint,
            "checkpoint_dir": resolved,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
        },
        dataset_id=config.dataset_id,
        dataset_hash=content_hash(load_reactions(config.dataset_dir)),
        notes="ReactionT5 fine-tuned with a frozen ESM-2 650M enzyme-sequence embedding "
        "prepended to the encoder.",
    )
    run_dir = evaluate_model(inputs=inputs, metadata=metadata, run_root=run_root)
    return str(run_dir)
