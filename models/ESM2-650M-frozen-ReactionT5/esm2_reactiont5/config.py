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
    init_checkpoint: str = ""  # if set, fine-tune from this saved wrapper dir instead of base
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
    num_beams: int = 10
    num_return_sequences: int = 10  # ranked predictions per reaction (<= num_beams)

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


# Continue-fine-tune the EnzymeMap-trained model on CYP-conditioned drug metabolism
# (Victorien-CYP-metabolites). ESM-2 stays frozen; only ReactionT5 + the projection
# update. Small dataset -> lower LR / few epochs; save only the final model.
FINETUNE_CONFIG = replace(
    TrainConfig(),
    dataset_dir="data/Victorien-CYP-metabolites/processed",
    dataset_id="victorien-cyp-metabolites-v1",
    init_checkpoint="models/ESM2-650M-frozen-ReactionT5/checkpoints-init",
    output_dir="models/ESM2-650M-frozen-ReactionT5/checkpoints-victorien-ft",
    learning_rate=5e-5,
    num_train_epochs=5.0,
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy="no",
)
