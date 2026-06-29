"""ReactionT5 conditioned on a *trainable* ESM-2 protein encoder.

ESM-2 embeds the enzyme sequence live on every forward pass; the mask-aware
mean-pooled (B, esm_dim) vector is projected to the T5 hidden size and prepended to
the encoder input embeddings as a single soft "enzyme token" (the attention mask is
extended by one). ReactionT5, the projection, **and ESM** are all fine-tuned — the
only structural difference from the frozen sibling is that ESM lives inside this
module (forward + backward + optimizer state) rather than as a precomputed tensor.

Gradient checkpointing on ESM is the main lever for fitting the O(L^2) protein
activations into memory; enable it via :meth:`from_base`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, EsmModel

from .esm import ESM_DIM, mean_pool

_PROJ_FILE = "protein_proj.pt"
_ESM_SUBDIR = "esm"


class ReactionT5WithTrainableProtein(nn.Module):
    def __init__(self, t5: Any, esm: Any, esm_dim: int = ESM_DIM) -> None:
        super().__init__()
        self.t5 = t5
        self.esm = esm
        self.protein_proj = nn.Linear(esm_dim, int(t5.config.d_model))

    @classmethod
    def from_base(
        cls,
        base_checkpoint: str,
        esm_model_id: str,
        esm_dim: int = ESM_DIM,
        *,
        gradient_checkpointing: bool = False,
    ) -> ReactionT5WithTrainableProtein:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(base_checkpoint)
        # No pooling head: we mean-pool ourselves, so the CLS pooler would be dead weight.
        esm = cast("Any", EsmModel).from_pretrained(esm_model_id, add_pooling_layer=False)
        model = cls(t5, esm, esm_dim)
        if gradient_checkpointing:
            model.enable_esm_gradient_checkpointing()
        return model

    @classmethod
    def from_pretrained_dir(
        cls, save_dir: str | Path, esm_dim: int = ESM_DIM
    ) -> ReactionT5WithTrainableProtein:
        save_dir = Path(save_dir)
        t5 = AutoModelForSeq2SeqLM.from_pretrained(save_dir)
        esm = cast("Any", EsmModel).from_pretrained(save_dir / _ESM_SUBDIR, add_pooling_layer=False)
        model = cls(t5, esm, esm_dim)
        model.protein_proj.load_state_dict(torch.load(save_dir / _PROJ_FILE, map_location="cpu"))
        return model

    def enable_esm_gradient_checkpointing(self) -> None:
        """Trade ~30% compute for a large drop in ESM activation memory.

        Uses non-reentrant checkpointing (no requires-grad-input constraint) and
        disables the KV cache, which is incompatible with checkpointing.
        """
        self.esm.config.use_cache = False
        self.esm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # save_pretrained handles T5's tied embeddings (which raw safetensors saving
        # of the wrapper state dict cannot); ESM goes in its own subdir.
        self.t5.save_pretrained(save_dir)
        self.esm.save_pretrained(save_dir / _ESM_SUBDIR)
        torch.save(self.protein_proj.state_dict(), save_dir / _PROJ_FILE)

    def _protein_token(self, esm_input_ids: Any, esm_attention_mask: Any) -> Any:
        hidden = self.esm(
            input_ids=esm_input_ids, attention_mask=esm_attention_mask
        ).last_hidden_state  # (B, L_esm, esm_dim)
        pooled = mean_pool(hidden, esm_attention_mask)  # (B, esm_dim)
        return self.protein_proj(pooled).unsqueeze(1)  # (B, 1, d)

    def _encoder_inputs(
        self, input_ids: Any, attention_mask: Any, esm_input_ids: Any, esm_attention_mask: Any
    ) -> Any:
        token_embeds = self.t5.get_input_embeddings()(input_ids)  # (B, L, d)
        protein_token = self._protein_token(esm_input_ids, esm_attention_mask)  # (B, 1, d)
        inputs_embeds = torch.cat([protein_token, token_embeds], dim=1)  # (B, 1+L, d)
        prefix = attention_mask.new_ones((attention_mask.shape[0], 1))
        mask = torch.cat([prefix, attention_mask], dim=1)
        return inputs_embeds, mask

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        esm_input_ids: Any = None,
        esm_attention_mask: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds, mask = self._encoder_inputs(
            input_ids, attention_mask, esm_input_ids, esm_attention_mask
        )
        return self.t5(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Any,
        attention_mask: Any,
        esm_input_ids: Any,
        esm_attention_mask: Any,
        **gen_kwargs: Any,
    ) -> Any:
        inputs_embeds, mask = self._encoder_inputs(
            input_ids, attention_mask, esm_input_ids, esm_attention_mask
        )
        encoder_outputs = self.t5.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=mask)
        return self.t5.generate(encoder_outputs=encoder_outputs, attention_mask=mask, **gen_kwargs)
