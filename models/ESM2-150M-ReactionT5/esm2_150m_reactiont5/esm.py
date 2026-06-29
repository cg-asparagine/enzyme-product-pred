"""ESM-2 150M encoder helpers for end-to-end (trainable) enzyme conditioning.

The frozen sibling model precomputes a fixed embedding per sequence and caches it
to ``.npz``. Here ESM is part of the training graph instead, so every forward pass
tokenizes and embeds the protein live — there is no cache. This module only holds
the small, model-agnostic pieces (device pick, the ESM model id/dim, mask-aware
mean-pool); the live ESM forward lives in :mod:`model`.

Sequences longer than the ESM context (1022 residues) are truncated before pooling.
"""

from __future__ import annotations

from typing import Any

ESM_MODEL_ID = "facebook/esm2_t30_150M_UR50D"
ESM_DIM = 640  # hidden size of esm2_t30_150M_UR50D
ESM_MAX_RESIDUES = 1022  # ESM-2 context is 1024 incl. <cls>/<eos>


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mask-aware mean over the sequence axis: (B, L, H), (B, L) -> (B, H).

    Differentiable: used inside the model's forward so ESM is trained end-to-end.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
