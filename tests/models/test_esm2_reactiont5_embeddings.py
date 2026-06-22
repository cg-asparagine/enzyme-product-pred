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


def test_ensure_embeddings_caches_and_subsets(tmp_path, monkeypatch):
    from esm2_reactiont5 import embeddings as emb

    seen: list[list[str]] = []

    def fake_embed(seqs, **_):
        seen.append(list(seqs))
        return np.stack([np.full(emb.ESM_DIM, float(len(s)), np.float32) for s in seqs])

    monkeypatch.setattr(emb, "embed_sequences", fake_embed)
    cache = tmp_path / "e.npz"

    out = emb.ensure_embeddings(cache, {"P1": "AAA", "P2": "AAAAA"})
    assert set(out) == {"P1", "P2"}
    assert np.allclose(out["P1"], 3.0)
    assert np.allclose(out["P2"], 5.0)

    seen.clear()
    out2 = emb.ensure_embeddings(cache, {"P1": "AAA", "P3": "AA"})
    assert set(out2) == {"P1", "P3"}  # returns only the requested subset
    assert seen == [["AA"]]  # only the uncached id is embedded


@pytest.mark.slow
@pytest.mark.network
def test_embed_sequences_real_esm():
    # Downloads ESM-2 650M (~2.5 GB); deselected from `just check`, run via test-slow.
    embeddings = embed_sequences(["MKTAYIAKQR", "MAAAAGGGTT"], batch_size=2)
    assert embeddings.shape == (2, ESM_DIM)
    assert np.isfinite(embeddings).all()
