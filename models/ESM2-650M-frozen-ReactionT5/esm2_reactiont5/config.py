"""Configuration for ESM2-650M-frozen-ReactionT5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass
class TrainConfig:
    # Data
    dataset_dir: str = "data/EnzymeMap_with_seq/processed"
    dataset_id: str = "enzymemap-with-seq-v1"
    embedding_cache: str = "models/ESM2-650M-frozen-ReactionT5/embeddings/esm2_t33_650m.npz"
    max_input_length: int = 256  # T5 tokens for "REACTANT:...REAGENT:"
    max_target_length: int = 200  # product-SMILES tokens
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Model
    base_checkpoint: str = "sagawa/ReactionT5v2-forward"
    esm_dim: int = 1280

    # Training
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
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
    output_dir: str = "models/ESM2-650M-frozen-ReactionT5/checkpoints"

    def as_dict(self) -> dict:
        return asdict(self)


# Tiny CPU/MPS-friendly config for the end-to-end smoke test.
SMOKE_CONFIG = replace(
    TrainConfig(),
    max_train_samples=16,
    max_eval_samples=8,
    max_input_length=128,
    max_target_length=64,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1.0,
    logging_steps=1,
    save_strategy="no",
    num_beams=2,
    num_return_sequences=2,
    device="cpu",  # tiny + portable; avoids contending with an MPS precompute
    output_dir="models/ESM2-650M-frozen-ReactionT5/checkpoints-smoke",
)

# The expanded dataset (curated + resolved accessions). See
# data/EnzymeMap_with_seq_plus/README.md.
PLUS_DATASET_DIR = "data/EnzymeMap_with_seq_plus/processed"
PLUS_DATASET_ID = "enzymemap-with-seq-plus-v1"


def with_plus_dataset(config: TrainConfig) -> TrainConfig:
    """Point a config at EnzymeMap_with_seq_plus with a distinct ``-plus`` checkpoint dir.

    Composes with ``SMOKE_CONFIG`` (-> ``checkpoints-smoke-plus``). The embedding cache
    is left unchanged on purpose: it is keyed by UniProt id, so the curated proteins
    already cached are reused and only the new (resolved) ones are embedded.
    """
    return replace(
        config,
        dataset_dir=PLUS_DATASET_DIR,
        dataset_id=PLUS_DATASET_ID,
        output_dir=config.output_dir + "-plus",
    )


# Full-size training on the expanded dataset (`just train ... --plus`).
PLUS_CONFIG = with_plus_dataset(TrainConfig())
