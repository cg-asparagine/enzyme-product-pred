# ESM2-150M-ReactionT5

Fine-tune **ReactionT5** to predict an enzymatic reaction's products from the
reactant SMILES, **conditioned on the enzyme sequence** via a **trainable ESM-2
150M** encoder. The end-to-end-trainable counterpart to
[`ESM2-650M-frozen-ReactionT5`](../ESM2-650M-frozen-ReactionT5/README.md): same
"one enzyme token prepended to the encoder" architecture, but ESM is fine-tuned
with the rest of the model instead of contributing a frozen, precomputed embedding.

- **Task:** generative — reactant-side SMILES → product-side SMILES.
- **Dataset:** [`EnzymeMap_with_seq`](../../data/EnzymeMap_with_seq/README.md).
- **Enzyme conditioning:** the enzyme sequence is tokenized and embedded by a
  **trainable** ESM-2 150M (`facebook/esm2_t30_150M_UR50D`) on every forward pass,
  mask-aware mean-pooled to a 640-d vector, projected to ReactionT5's hidden size,
  and **prepended to the encoder as a single soft "enzyme token"**. ReactionT5, the
  projection, **and ESM** are all fine-tuned.
- **Backbone:** `sagawa/ReactionT5v2-forward`; input format
  `REACTANT:{reactants}REAGENT:` (empty reagent block).

## Why 150M (and not 650M trainable)

Training target is an **Apple M4 Pro, 48 GB unified memory, MPS, fp32** (no
fp16/bf16 autocast). A trainable ESM lives in the graph — weights + grads + Adam
state (16 bytes/param) + O(L²) activations over long enzymes (median ~400, p99
~1300 residues) — so size is memory-bound:

| ESM-2 | weights+grad+Adam | + T5 (~120M) | on 48 GB / MPS |
|-------|-------------------|--------------|----------------|
| 35M   | 0.6 GB            | ~2.5 GB      | trivial |
| **150M** | **2.4 GB**     | **~4.3 GB**  | **comfortable — batch 4–8** |
| 650M  | 10.4 GB           | ~12.3 GB     | the ceiling: grad-checkpoint + batch 1–2, slow |
| 3B    | 48 GB             | >50 GB       | ✗ optimizer state alone exceeds RAM |

150M is the largest that trains comfortably here. The frozen sibling can use a much
larger ESM precisely *because* it never backprops through it.

## Run

```
just train ESM2-150M-ReactionT5       # fine-tune end-to-end (downloads ESM-2 150M)
just evaluate ESM2-150M-ReactionT5    # -> experiments/<run_id>/report.pdf
```

Append `--smoke` for a tiny CPU run (uses the 8M ESM). Append `--plus` to
train/evaluate on [`EnzymeMap_with_seq_plus`](../../data/EnzymeMap_with_seq_plus/README.md)
(checkpoints go to `checkpoints-plus`; the curated model is untouched). For the
honest **new-enzyme** comparison, evaluate on the enzyme-cluster split (`enzyme_split`).

## Memory levers (config.py)

- `gradient_checkpointing` (default **True**) — trades ~30% compute for a large drop
  in ESM activation memory; what lets full-length proteins fit on MPS/48 GB.
- `max_residues` (default 1022) — protein truncation; ESM attention is O(L²), so
  lowering this (e.g. 512) is the cheapest way to cut memory if you disable
  checkpointing for speed.
- `per_device_train_batch_size` (4) × `gradient_accumulation_steps` (4) → effective 16.
- `esm_learning_rate` (2e-5) vs `learning_rate` (1e-4) — the pretrained ESM encoder
  gets a smaller LR than the task-side params (T5 + projection) to limit forgetting.

## Layout

- `esm2_150m_reactiont5/esm.py` — ESM constants + mask-aware `mean_pool` (no cache).
- `esm2_150m_reactiont5/model.py` — ReactionT5 + **trainable** ESM + projection.
- `esm2_150m_reactiont5/data.py` — dataset (ESM-tokenizes the sequence) + collator.
- `esm2_150m_reactiont5/{config,pipeline}.py` — config (+ `SMOKE_CONFIG`) and train/evaluate.
- `train.py` / `evaluate.py` — thin entrypoints.

## Saved-model layout

`save()` writes a self-contained, directly-evaluable dir: ReactionT5 via
`save_pretrained` (tied-embedding-safe) + the T5 tokenizer at the root, the
fine-tuned ESM + its tokenizer under `esm/`, and the projection as `protein_proj.pt`.
(Trainer resume-from-checkpoint of the wrapper is not supported.)

## Notes / caveats

- ReactionT5 was pretrained at 150 input tokens; long cofactor-heavy reactions
  exceed that. T5's relative attention extrapolates, but quality may drop — tune
  `max_input_length`.
- ESM-2's context is 1022 residues; longer enzymes are truncated before pooling.
- `device` defaults to `auto` (cuda → mps → cpu); the smoke config forces `cpu`.
- On MPS, end-to-end ESM fine-tuning is markedly slower than the frozen model (which
  pays the ESM forward once, offline). Expect hours, not minutes, per epoch.
