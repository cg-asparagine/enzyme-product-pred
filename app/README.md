# Model-explorer GUI

A local Streamlit app for inspecting one reaction at a time — its **enzyme**, its
**substrate** structures, and a trained model's **predicted products** beside the
ground truth. Launch it with:

```
just gui            # = uv run streamlit run app/streamlit_app.py
```

## What it shows

Pick a model in the sidebar, then choose a reaction:

- **Dataset split** — browse the `test` / `valid` / `train` split of the model's
  dataset (under either the reaction `split` or the new-enzyme `enzyme_split`).
  Filter by EC number, organism, or UniProt id.
- **Custom reaction** — paste reactant SMILES and condition on either a dataset
  enzyme (reuses its precomputed ESM-2 embedding — fast) or a pasted protein
  sequence (embedded live with frozen ESM-2 650M — a heavy one-off load).

For the chosen reaction you see the **enzyme & conditions** (EC number, organism,
UniProt id, direction, full sequence), the **substrate** molecules, and the
**ground-truth products** (with the changed region highlighted vs. the substrate).
Hit **Predict** to run the model and render its ranked product candidates, each
badged with an ✅ exact-set match or its Tanimoto similarity to the nearest true
product, plus a recovered-products tally.

## Layout

- `streamlit_app.py` — the UI (the only file Streamlit runs). Model/source
  selection, the enzyme/substrate/product views, and the prediction grid.
- `registry.py` — model + dataset specs, dataset→`ReactionEntry` building,
  prediction↔truth matching, and the lazy-loading `Adapter` that wraps a trained
  model behind a uniform `predict`. Heavy ML imports happen only in
  `load_adapter`, so importing this module is cheap (and test-safe).
- `render.py` — pure, cached RDKit SMILES→PNG rendering, with change-vs-substrate
  highlighting via maximum common substructure.

Models load on first Predict and are cached for the session. The app reads
trained checkpoints from `models/<Model>/checkpoints/` and the ESM-2 embedding
cache; it is not used by training or the eval framework.
