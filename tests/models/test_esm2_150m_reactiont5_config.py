"""Config presets for the trainable model: --plus dataset + the tiny smoke ESM."""

from esm2_150m_reactiont5.config import (
    PLUS_CONFIG,
    PLUS_DATASET_DIR,
    PLUS_DATASET_ID,
    SMOKE_CONFIG,
    TrainConfig,
    with_plus_dataset,
)


def test_plus_config_points_at_expanded_dataset():
    base = TrainConfig()
    assert PLUS_CONFIG.dataset_dir == PLUS_DATASET_DIR
    assert PLUS_CONFIG.dataset_id == PLUS_DATASET_ID
    # Distinct checkpoint dir so it never clobbers the curated model.
    assert PLUS_CONFIG.output_dir == base.output_dir + "-plus"
    assert PLUS_CONFIG.output_dir != base.output_dir


def test_with_plus_dataset_composes_with_smoke():
    smoke_plus = with_plus_dataset(SMOKE_CONFIG)
    assert smoke_plus.dataset_dir == PLUS_DATASET_DIR
    assert smoke_plus.output_dir == "models/ESM2-150M-ReactionT5/checkpoints-smoke-plus"
    # Smoke shrink is preserved (only the dataset/output change).
    assert smoke_plus.max_train_samples == SMOKE_CONFIG.max_train_samples
    assert smoke_plus.device == "cpu"


def test_default_is_trainable_150m_smoke_is_tiny_8m():
    base = TrainConfig()
    assert base.esm_model_id == "facebook/esm2_t30_150M_UR50D"
    assert base.esm_dim == 640
    assert base.gradient_checkpointing is True
    # Smoke swaps in the 8M ESM (dim 320) so the end-to-end test stays tiny.
    assert SMOKE_CONFIG.esm_model_id == "facebook/esm2_t6_8M_UR50D"
    assert SMOKE_CONFIG.esm_dim == 320
    assert SMOKE_CONFIG.gradient_checkpointing is False
