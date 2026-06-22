# ESM2-650M-frozen-ReactionT5

Fine-tune **ReactionT5** to predict an enzymatic reaction's products from the
reactant SMILES, **conditioned on the enzyme sequence** via a frozen **ESM-2 650M**
embedding.

- **Task:** generative — reactant-side SMILES → product-side SMILES.
- **Dataset:** [`EnzymeMap_with_seq`](../../data/EnzymeMap_with_seq/README.md).
- **Enzyme conditioning:** each unique sequence is embedded once with frozen
  ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), mean-pooled to a 1280-d vector and
  cached (`embeddings/`, git-ignored). At train/eval time the vector is projected
  to ReactionT5's hidden size and **prepended to the encoder as a single soft
  "enzyme token"**. ESM is never fine-tuned; ReactionT5 + the projection are.
- **Backbone:** `sagawa/ReactionT5v2-forward`; input format
  `REACTANT:{reactants}REAGENT:` (empty reagent block).

## Run

```
just train ESM2-650M-frozen-ReactionT5       # fine-tune (auto-embeds needed sequences on first run)
just evaluate ESM2-650M-frozen-ReactionT5    # -> experiments/<run_id>/report.pdf
```

Append `--smoke` for a tiny CPU run. To precompute **all** embeddings up front
(recommended; resumable, ~15k sequences):

```
uv run python -c "import sys; sys.path.insert(0, 'models/ESM2-650M-frozen-ReactionT5'); \
from esm2_reactiont5.embeddings import precompute_embeddings; \
precompute_embeddings('data/EnzymeMap_with_seq/processed', \
'models/ESM2-650M-frozen-ReactionT5/embeddings/esm2_t33_650m.npz')"
```

## Layout

- `esm2_reactiont5/embeddings.py` — frozen ESM-2 embedding + `.npz` cache.
- `esm2_reactiont5/model.py` — ReactionT5 + protein projection (encoder soft token).
- `esm2_reactiont5/data.py` — dataset + collator.
- `esm2_reactiont5/{config,pipeline}.py` — config (+ `SMOKE_CONFIG`) and train/evaluate.
- `train.py` / `evaluate.py` — thin entrypoints.

## Notes / caveats

- ReactionT5 was pretrained at 150 input tokens; long cofactor-heavy reactions
  exceed that. T5's relative attention extrapolates, but quality may drop — tune
  `max_input_length` in `config.py`.
- ESM-2's context is 1022 residues; longer enzymes are truncated before pooling.
- `device` defaults to `auto` (cuda → mps → cpu); the smoke config forces `cpu`.
- The 650M embeddings are frozen, so this never fine-tunes ESM. Upgrade paths:
  larger frozen ESM (3B) embeddings, or unfreeze a smaller ESM for end-to-end
  fine-tuning.
