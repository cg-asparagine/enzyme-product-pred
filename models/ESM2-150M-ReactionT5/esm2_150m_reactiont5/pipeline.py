"""Train and evaluate ESM2-150M-ReactionT5 (trainable ESM-2, fine-tuned end-to-end)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from epp_core.chem.tokenize import detokenize
from epp_core.data import content_hash, load_reactions, train_product_smiles
from epp_core.eval.types import GenerativeEvalInputs, TaskType
from epp_core.metadata.capture import ExperimentMetadata
from epp_core.runner import evaluate_model

from .config import TrainConfig
from .data import ProteinSeq2SeqCollator, ReactionSequenceDataset, format_input
from .esm import select_device
from .model import ReactionT5WithTrainableProtein

_MODEL_NAME = "ESM2-150M-ReactionT5"


def _subset(frame: Any, n: int | None) -> Any:
    return frame if n is None else frame.head(n)


def _resolve_device(config: TrainConfig) -> str:
    return select_device() if config.device == "auto" else config.device


def _esm_param_groups(model: Any, args: Any, esm_lr: float, base_lr: float) -> list[dict]:
    """Two learning rates: a small one for the pretrained ESM encoder, the base one
    for the task-side params (T5 + projection). Weight-decay split mirrors the stock
    HF Trainer (no decay on LayerNorm/bias), so these groups differ from default only
    in the per-group LR.
    """
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names

    decay = set(get_parameter_names(model, list(ALL_LAYERNORM_LAYERS)))
    decay = {n for n in decay if "bias" not in n}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def group(is_esm: bool, do_decay: bool) -> list[Any]:
        return [p for n, p in named if n.startswith("esm.") == is_esm and (n in decay) == do_decay]

    wd = args.weight_decay
    candidates = [
        {"params": group(True, True), "weight_decay": wd, "lr": esm_lr},
        {"params": group(True, False), "weight_decay": 0.0, "lr": esm_lr},
        {"params": group(False, True), "weight_decay": wd, "lr": base_lr},
        {"params": group(False, False), "weight_decay": 0.0, "lr": base_lr},
    ]
    return [g for g in candidates if g["params"]]


def _shuffle_enzyme_conditioning(frame: Any, seed: int) -> Any:
    """Control: permute the ``(uniprot_id, sequence)`` enzyme pair across rows.

    Breaks the substrate->enzyme correspondence (each reaction is conditioned on
    some other row's enzyme) while keeping every id<->sequence pair intact. Reactants
    and reference products are left in place, so a score drop vs. the unshuffled run
    isolates the enzyme's effect.
    """
    pairs = list(zip(frame["uniprot_id"].tolist(), frame["sequence"].tolist(), strict=True))
    order = np.random.default_rng(seed).permutation(len(pairs))
    frame = frame.copy()
    frame["uniprot_id"] = [pairs[i][0] for i in order]
    frame["sequence"] = [pairs[i][1] for i in order]
    return frame


def train_model(config: TrainConfig) -> str:
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    class _ProteinTrainer(Trainer):
        """Differential-LR optimizer (low LR for ESM) + tied-weight-safe checkpoints.

        ``create_optimizer`` builds ESM/task param groups (see :func:`_esm_param_groups`).
        ``_save`` routes through ``model.save`` so T5's tied embeddings are written via
        ``save_pretrained`` (raw safetensors saving of the wrapper state dict raises on
        them) and ESM lands in its own subdir — every epoch checkpoint is a complete,
        directly-evaluable model. (Trainer resume-from-checkpoint is not supported.)
        """

        def create_optimizer(self, model: Any = None) -> Any:
            if self.optimizer is None:
                groups = _esm_param_groups(
                    model or self.model, self.args, config.esm_learning_rate, config.learning_rate
                )
                self.optimizer = torch.optim.AdamW(
                    groups,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return self.optimizer

        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            out = Path(cast(str, output_dir or self.args.output_dir))
            cast(ReactionT5WithTrainableProtein, self.accelerator.unwrap_model(self.model)).save(
                out
            )
            if self.processing_class is not None:
                self.processing_class.save_pretrained(out)
            esm_tokenizer.save_pretrained(out / "esm")
            torch.save(self.args, out / "training_args.bin")

    device = _resolve_device(config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_checkpoint)
    esm_tokenizer = AutoTokenizer.from_pretrained(config.esm_model_id)
    train_df = _subset(load_reactions(config.dataset_dir, "train"), config.max_train_samples)

    train_ds = ReactionSequenceDataset(
        train_df,
        tokenizer,
        esm_tokenizer,
        config.max_input_length,
        config.max_target_length,
        config.max_residues,
    )
    collator = ProteinSeq2SeqCollator(tokenizer, esm_tokenizer)
    model = ReactionT5WithTrainableProtein.from_base(
        config.base_checkpoint,
        config.esm_model_id,
        config.esm_dim,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
    trainer = _ProteinTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    save_dir = Path(config.output_dir)
    model.save(save_dir)
    tokenizer.save_pretrained(save_dir)
    esm_tokenizer.save_pretrained(save_dir / "esm")
    return str(save_dir)


def _generate(
    model: ReactionT5WithTrainableProtein,
    tokenizer: Any,
    esm_tokenizer: Any,
    frame: Any,
    config: TrainConfig,
    device: str,
) -> list[list[str]]:
    reactants = frame["reactant_smiles"].tolist()
    sequences = frame["sequence"].tolist()
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
        seqs = [str(s)[: config.max_residues] for s in sequences[start : start + batch]]
        esm_enc = esm_tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_residues + 2,
        ).to(device)
        generated = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            esm_input_ids=esm_enc["input_ids"],
            esm_attention_mask=esm_enc["attention_mask"],
            num_beams=config.num_beams,
            num_return_sequences=k,
            max_length=config.max_target_length,
        )
        # ReactionT5's tokenizer can emit internal spaces when decoding; a space
        # terminates a SMILES for RDKit, so strip whitespace before scoring.
        decoded = [
            detokenize(s) for s in tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]
        for i in range(len(srcs)):
            predictions.append(decoded[i * k : (i + 1) * k])
    return predictions


def evaluate_run(
    config: TrainConfig,
    model_dir: str | None = None,
    run_root: str = "experiments",
    shuffle_enzymes: bool = False,
) -> str:
    from transformers import AutoTokenizer

    resolved = model_dir or config.output_dir
    device = _resolve_device(config)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    esm_tokenizer = AutoTokenizer.from_pretrained(str(Path(resolved) / "esm"))
    model = ReactionT5WithTrainableProtein.from_pretrained_dir(resolved, config.esm_dim)
    model.to(device)
    model.eval()

    test_df = _subset(load_reactions(config.dataset_dir, "test"), config.max_eval_samples)
    if shuffle_enzymes:  # control: see whether conditioning on the correct enzyme helps
        test_df = _shuffle_enzyme_conditioning(test_df, seed=config.seed)
    references = [str(p).split(".") for p in test_df["product_smiles"]]
    predictions = _generate(model, tokenizer, esm_tokenizer, test_df, config, device)
    inputs = GenerativeEvalInputs(
        references=references,
        predictions=predictions,
        train_smiles=train_product_smiles(config.dataset_dir),
    )

    architecture = {
        "base_checkpoint": config.base_checkpoint,
        "esm_model": config.esm_model_id,
        "esm_frozen": False,
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
            "esm_model": config.esm_model_id,
            "checkpoint_dir": resolved,
            "learning_rate": config.learning_rate,
            "esm_learning_rate": config.esm_learning_rate,
            "gradient_checkpointing": config.gradient_checkpointing,
            "num_train_epochs": config.num_train_epochs,
        },
        dataset_id=config.dataset_id + ("-shuffled-enzymes" if shuffle_enzymes else ""),
        dataset_hash=content_hash(load_reactions(config.dataset_dir)),
        notes="ReactionT5 fine-tuned end-to-end with a TRAINABLE ESM-2 150M enzyme-sequence "
        "encoder (mean-pooled, prepended to the encoder as one soft token)."
        + (
            " CONTROL: enzyme conditioning permuted across test rows (fixed seed) to "
            "measure whether the correct enzyme helps."
            if shuffle_enzymes
            else ""
        ),
    )
    run_dir = evaluate_model(inputs=inputs, metadata=metadata, run_root=run_root)
    return str(run_dir)
