"""Configuration for ESM2-150M-ReactionT5 (trainable ESM-2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .esm import ESM_DIM, ESM_MAX_RESIDUES, ESM_MODEL_ID


@dataclass
class TrainConfig:
    # Data
    dataset_dir: str = "data/EnzymeMap_with_seq/processed"
    dataset_id: str = "enzymemap-with-seq-v1"
    max_input_length: int = 256  # T5 tokens for "REACTANT:...REAGENT:"
    max_target_length: int = 200  # product-SMILES tokens
    max_residues: int = ESM_MAX_RESIDUES  # protein truncation; bounds ESM's O(L^2) cost
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Model
    base_checkpoint: str = "sagawa/ReactionT5v2-forward"
    esm_model_id: str = ESM_MODEL_ID
    esm_dim: int = ESM_DIM
    # ESM is in the training graph: checkpointing trades ~30% compute for a large
    # activation-memory drop, which is what makes a trainable ESM fit on MPS/48 GB.
    gradient_checkpointing: bool = True

    # Training
    learning_rate: float = 1e-4  # T5 + projection (task-side params)
    esm_learning_rate: float = 2e-5  # smaller LR for the pretrained ESM encoder
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch 16
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_strategy: str = "epoch"
    seed: int = 42
    fp16: bool = False  # MPS lacks fp16 autocast; keep off
    device: str = "auto"  # "auto" -> cuda/mps/cpu; force with "cpu"/"mps"/"cuda"

    # Generation (eval)
    num_beams: int = 5
    num_return_sequences: int = 5  # ranked predictions per reaction (<= num_beams)

    # IO
    output_dir: str = "models/ESM2-150M-ReactionT5/checkpoints"

    def as_dict(self) -> dict:
        return asdict(self)


# Tiny CPU-friendly config for the end-to-end smoke test. Uses the 8M ESM (already a
# common local cache) so the smoke run trains a real — but tiny — end-to-end graph.
SMOKE_CONFIG = replace(
    TrainConfig(),
    esm_model_id="facebook/esm2_t6_8M_UR50D",
    esm_dim=320,
    max_residues=64,
    max_train_samples=16,
    max_eval_samples=8,
    max_input_length=128,
    max_target_length=64,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,  # tiny model + tiny seqs: checkpointing only slows it
    num_train_epochs=1.0,
    logging_steps=1,
    save_strategy="no",
    num_beams=2,
    num_return_sequences=2,
    device="cpu",  # tiny + portable
    output_dir="models/ESM2-150M-ReactionT5/checkpoints-smoke",
)

# The expanded dataset (curated + resolved accessions). See
# data/EnzymeMap_with_seq_plus/README.md.
PLUS_DATASET_DIR = "data/EnzymeMap_with_seq_plus/processed"
PLUS_DATASET_ID = "enzymemap-with-seq-plus-v1"


def with_plus_dataset(config: TrainConfig) -> TrainConfig:
    """Point a config at EnzymeMap_with_seq_plus with a distinct ``-plus`` checkpoint dir.

    Composes with ``SMOKE_CONFIG`` (-> ``checkpoints-smoke-plus``). Unlike the frozen
    model there is no embedding cache to share — ESM is embedded live each step.
    """
    return replace(
        config,
        dataset_dir=PLUS_DATASET_DIR,
        dataset_id=PLUS_DATASET_ID,
        output_dir=config.output_dir + "-plus",
    )


# Full-size training on the expanded dataset (`just train ... --plus`).
PLUS_CONFIG = with_plus_dataset(TrainConfig())
