"""ReactionT5 conditioned on a frozen ESM-2 protein embedding.

The 1280-d protein embedding is projected to the T5 hidden size and prepended to
the encoder input embeddings as a single soft "enzyme token"; the attention mask
is extended by one. ReactionT5's weights and the projection are fine-tuned; ESM
stays frozen (its embeddings are precomputed and passed in as tensors, never part
of this module's graph).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

_PROJ_FILE = "protein_proj.pt"


class ReactionT5WithProtein(nn.Module):
    def __init__(self, t5: Any, esm_dim: int = 1280) -> None:
        super().__init__()
        self.t5 = t5
        self.protein_proj = nn.Linear(esm_dim, int(t5.config.d_model))

    @classmethod
    def from_base(cls, base_checkpoint: str, esm_dim: int = 1280) -> ReactionT5WithProtein:
        return cls(AutoModelForSeq2SeqLM.from_pretrained(base_checkpoint), esm_dim)

    @classmethod
    def from_pretrained_dir(
        cls, save_dir: str | Path, esm_dim: int = 1280
    ) -> ReactionT5WithProtein:
        save_dir = Path(save_dir)
        model = cls(AutoModelForSeq2SeqLM.from_pretrained(save_dir), esm_dim)
        model.protein_proj.load_state_dict(torch.load(save_dir / _PROJ_FILE, map_location="cpu"))
        return model

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.t5.save_pretrained(save_dir)
        torch.save(self.protein_proj.state_dict(), save_dir / _PROJ_FILE)

    def _encoder_inputs(self, input_ids: Any, attention_mask: Any, protein_embedding: Any) -> Any:
        token_embeds = self.t5.get_input_embeddings()(input_ids)  # (B, L, d)
        protein_token = self.protein_proj(protein_embedding).unsqueeze(1)  # (B, 1, d)
        inputs_embeds = torch.cat([protein_token, token_embeds], dim=1)  # (B, 1+L, d)
        prefix = attention_mask.new_ones((attention_mask.shape[0], 1))
        mask = torch.cat([prefix, attention_mask], dim=1)
        return inputs_embeds, mask

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        protein_embedding: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds, mask = self._encoder_inputs(input_ids, attention_mask, protein_embedding)
        return self.t5(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(
        self, input_ids: Any, attention_mask: Any, protein_embedding: Any, **gen_kwargs: Any
    ) -> Any:
        inputs_embeds, mask = self._encoder_inputs(input_ids, attention_mask, protein_embedding)
        encoder_outputs = self.t5.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=mask)
        return self.t5.generate(encoder_outputs=encoder_outputs, attention_mask=mask, **gen_kwargs)
