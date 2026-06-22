"""Frozen ESM-2 650M protein embeddings, precomputed and cached per UniProt id.

Each unique enzyme sequence is embedded once with a frozen ESM-2 650M encoder
(mean-pooled over residues -> a 1280-d vector) and cached to a ``.npz``, so
training the reaction model never pays the ESM forward cost and can use a much
larger ESM than end-to-end fine-tuning would allow. Sequences longer than the
ESM context (1022 residues) are truncated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

ESM_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ESM_DIM = 1280
ESM_MAX_RESIDUES = 1022  # ESM-2 context is 1024 incl. <cls>/<eos>
_CHUNK = 512  # embeddings flushed to the cache every this many sequences


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mask-aware mean over the sequence axis: (B, L, H), (B, L) -> (B, H)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def embed_sequences(
    sequences: list[str],
    *,
    batch_size: int = 8,
    max_residues: int = ESM_MAX_RESIDUES,
    device: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> np.ndarray:
    """Return an ``(N, ESM_DIM)`` float32 array of mean-pooled ESM-2 embeddings.

    Pass a preloaded ``model``/``tokenizer`` to avoid reloading across chunks (or
    to inject fakes in tests). Sequences are processed in length-sorted order for
    padding efficiency, then restored to input order.
    """
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = device or select_device()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_ID)
    if model is None:
        model = cast("Any", EsmModel.from_pretrained(ESM_MODEL_ID)).to(device).eval()

    out = np.zeros((len(sequences), ESM_DIM), dtype=np.float32)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = [sequences[i][:max_residues] for i in idx]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_residues + 2,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            pooled = mean_pool(hidden, enc["attention_mask"]).float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = pooled[j]
    return out


def load_embeddings(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Load the cached ``{uniprot_id: vector}`` mapping (empty if no cache yet)."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    data = np.load(cache_path, allow_pickle=False)
    return {str(i): e for i, e in zip(data["ids"], data["embeddings"], strict=True)}


def _save_embeddings(cache_path: Path, mapping: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(mapping.keys()))
    if mapping:
        embeddings = np.stack(list(mapping.values())).astype(np.float32)
    else:
        embeddings = np.zeros((0, ESM_DIM), dtype=np.float32)
    np.savez(cache_path, ids=ids, embeddings=embeddings)


def precompute_embeddings(
    processed_dir: str | Path,
    cache_path: str | Path,
    *,
    batch_size: int = 8,
    max_residues: int = ESM_MAX_RESIDUES,
) -> dict[str, np.ndarray]:
    """Embed every unique ``(uniprot_id, sequence)`` in a processed reactions
    dataset that isn't already cached, writing the ``.npz`` cache incrementally
    (resumable). Returns the full ``{uniprot_id: vector}`` mapping.
    """
    import pandas as pd

    cache_path = Path(cache_path)
    frame = pd.read_parquet(
        Path(processed_dir) / "reactions.parquet", columns=["uniprot_id", "sequence"]
    )
    by_id = dict(zip(frame["uniprot_id"], frame["sequence"], strict=True))  # 1:1 id -> sequence
    cache = load_embeddings(cache_path)
    todo = [u for u in by_id if u not in cache]
    if not todo:
        return cache

    from transformers import AutoTokenizer, EsmModel

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_ID)
    model = cast("Any", EsmModel.from_pretrained(ESM_MODEL_ID)).to(device).eval()

    for start in range(0, len(todo), _CHUNK):
        chunk = todo[start : start + _CHUNK]
        vectors = embed_sequences(
            [by_id[u] for u in chunk],
            batch_size=batch_size,
            max_residues=max_residues,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
        for u, vector in zip(chunk, vectors, strict=True):
            cache[u] = vector
        _save_embeddings(cache_path, cache)
        print(f"  embedded {min(start + _CHUNK, len(todo))}/{len(todo)} new sequences")
    return cache


def ensure_embeddings(
    cache_path: str | Path,
    id_to_seq: dict[str, str],
    *,
    batch_size: int = 8,
    max_residues: int = ESM_MAX_RESIDUES,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Return ``{id: vector}`` for exactly ``id_to_seq``, embedding (and caching)
    only the ids not already cached.

    Lets a training/eval run embed just the sequences it needs (e.g. a smoke
    test's handful) while reusing the full precomputed cache when present. To
    build the whole cache up front, prefer :func:`precompute_embeddings`
    (chunked + resumable).
    """
    cache_path = Path(cache_path)
    cache = load_embeddings(cache_path)
    todo = [u for u in id_to_seq if u not in cache]
    if todo:
        vectors = embed_sequences(
            [id_to_seq[u] for u in todo],
            batch_size=batch_size,
            max_residues=max_residues,
            device=device,
        )
        for uid, vector in zip(todo, vectors, strict=True):
            cache[uid] = vector
        _save_embeddings(cache_path, cache)
    return {uid: cache[uid] for uid in id_to_seq}
