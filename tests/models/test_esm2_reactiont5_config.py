"""The --plus config preset: points at EnzymeMap_with_seq_plus, reuses the cache."""

from esm2_reactiont5.config import (
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
    # Embedding cache is shared on purpose (keyed by UniProt id -> curated reuse).
    assert PLUS_CONFIG.embedding_cache == base.embedding_cache


def test_with_plus_dataset_composes_with_smoke():
    smoke_plus = with_plus_dataset(SMOKE_CONFIG)
    assert smoke_plus.dataset_dir == PLUS_DATASET_DIR
    assert smoke_plus.output_dir == "models/ESM2-650M-frozen-ReactionT5/checkpoints-smoke-plus"
    # Smoke shrink is preserved (only the dataset/output change).
    assert smoke_plus.max_train_samples == SMOKE_CONFIG.max_train_samples
    assert smoke_plus.device == "cpu"
