# enzyme-product-pred

Predict the **products of an enzymatic reaction** from its reactants and the
enzyme/conditions, using reaction data from **EnzymeMap v2 / BRENDA 2023**.
Multiple modelling approaches share one dataset folder and one **task-aware
evaluation framework** (`epp_core`) that records each run's full metadata +
hyperparameters and renders a **PDF report** of every metric and plot.

## Layout

- `src/epp_core/` — shared library: SMILES/reaction cheminformatics utils, the
  data pipeline, experiment-metadata capture, the metrics + plots + PDF-report
  engine, and the `evaluate_model` runner.
- `data/<Dataset>/` — one folder per dataset (`raw/` + `prepare.py` →
  `processed/`).
- `models/<Model>/` — one folder per model / framework.
- `experiments/<run_id>/` — per-run artifacts: `metadata.json`, `metrics.json`,
  `plots/`, `report.pdf`.

## Task

Generative reaction-product prediction: reactant-side SMILES (substrate +
cofactors) → product-side SMILES, optionally conditioned on EC number + organism.

## Datasets

- `EnzymeMap` — enzymatic reactions from EnzymeMap v2 / BRENDA 2023
  (`reactants>>products` SMILES + EC number, organism, quality). The 386 MB raw
  CSV is git-ignored; see [`data/EnzymeMap/README.md`](data/EnzymeMap/README.md).

## Quickstart

```
just install                 # uv sync (Python 3.12)
just data-prep EnzymeMap     # build data/EnzymeMap/processed/
just check                   # format + lint + typecheck + fast tests
```

Train/evaluate recipes (`just train <Model>` / `just evaluate <Model>`) land as
models are added; `just evaluate` writes a PDF report to `experiments/<run_id>/`.

## Contributing

Each dataset and model drops into its own folder under `data/` or `models/`;
shared, reusable code lives in `epp_core`. See [CLAUDE.md](CLAUDE.md) for the
contributor guide.
