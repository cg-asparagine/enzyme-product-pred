import numpy as np
import pytest
from esm2_reactiont5.embeddings import (
    ESM_DIM,
    _save_embeddings,
    embed_sequences,
    load_embeddings,
    mean_pool,
)


def test_mean_pool_ignores_padding():
    import torch

    # 1 example, 3 positions, hidden 2; the 3rd position is padding (mask 0).
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))  # mean of first two only


def test_load_embeddings_roundtrip(tmp_path):
    cache = tmp_path / "emb.npz"
    mapping = {"P1": np.ones(ESM_DIM, np.float32), "P2": np.zeros(ESM_DIM, np.float32)}
    _save_embeddings(cache, mapping)
    loaded = load_embeddings(cache)
    assert set(loaded) == {"P1", "P2"}
    assert np.allclose(loaded["P1"], 1.0)
    assert np.allclose(loaded["P2"], 0.0)


def test_load_embeddings_missing_returns_empty(tmp_path):
    assert load_embeddings(tmp_path / "nope.npz") == {}


@pytest.mark.slow
@pytest.mark.network
def test_embed_sequences_real_esm():
    # Downloads ESM-2 650M (~2.5 GB); deselected from `just check`, run via test-slow.
    embeddings = embed_sequences(["MKTAYIAKQR", "MAAAAGGGTT"], batch_size=2)
    assert embeddings.shape == (2, ESM_DIM)
    assert np.isfinite(embeddings).all()
