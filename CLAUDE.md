# CLAUDE.md

Guidance for working in this repository.

## Project

Predict the **products of an enzymatic reaction** from its reactants (substrate +
cofactors) and the enzyme/conditions. Reactions come from **EnzymeMap v2 /
BRENDA 2023** as reaction SMILES (`reactants>>products`). The first task framing
is **generative**: reactant-side SMILES → product-side SMILES, optionally
conditioned on the **EC number** and **organism** (the dataset has no amino-acid
sequences yet — see Roadmap).

Multiple modelling approaches share one `data/` folder and one **task-aware
evaluation framework** (`epp_core`) that records full model metadata +
hyperparameters for every run and renders a **PDF report** of all metrics and
plots.

## Repository layout

```
src/epp_core/            # SHARED library — all general, reusable code lives here
  chem/                  #   SMILES utils (canonicalize/validity/tanimoto), atom tokenizer,
                         #   reaction-SMILES helpers (split reactants>>products)
  data/                  #   readers, deterministic build_reactions(), loaders, splits, hashing
  metadata/              #   ExperimentMetadata (git/libs/device capture)
  eval/                  #   TaskType registry + generative metrics (top-k, validity, exact-set-match…)
  plots/  report/        #   matplotlib figures + ReportLab PDF
  runner.py              #   evaluate_model(): metadata -> metrics -> plots -> PDF
  io.py                  #   JSON I/O, run-dir naming
data/<Dataset>/          # ONE folder per dataset: raw/ (git-ignored if large), processed/ (ignored), prepare.py
models/<Model>/          # ONE folder per model / framework (added as we build them)
experiments/<run_id>/    # run artifacts (git-ignored): metadata.json, metrics.json, plots/, report.pdf
tests/                   # unit/ (fast, pure logic); model smoke tests are @slow
```

**Convention:** datasets and model/frameworks each get their own folder so new
ones drop in independently. **General, reusable code goes in `epp_core`, never in
a model or dataset folder.**

## Setup

- Python **3.12** via [`uv`](https://docs.astral.sh/uv/); task runner is
  [`just`](https://just.systems) (install separately, e.g. `brew install just`).
- `just install` (= `uv sync`) creates `.venv` with everything.

## Running

```
just data-prep <Dataset>   # build data/<Dataset>/processed/   (e.g. EnzymeMap)
just train <Model>         # train a model               (once a model exists)
just evaluate <Model>      # -> experiments/<run_id>/report.pdf
just report <run_dir>      # re-render a PDF from saved JSON
just data-report <Dataset> # dataset/split EDA PDF (clusters, splits, distributions)
just check                 # format + lint + typecheck + fast tests (the gate)
just test-slow             # @slow tests (real training; may download checkpoints)
just save "<msg>"          # run check, then git add + commit + push
```

## Adding a dataset

Create `data/<Name>/` with `raw/`, a `prepare.py` that reads raw files into
record dicts and calls `epp_core.data.build_reactions(...)` (writing
`processed/reactions.parquet` + `build_manifest.json`), and a `README.md` with
provenance + schema. Reuse `epp_core.data.readers` / `epp_core.chem`. Assign the
train/valid/test split **grouped on `rxn_idx`** so a reaction's forward/reverse
twins never cross splits.

## Adding a model / framework

Create `models/<Name>/` as a **script directory** (not an installed package;
hyphens allowed) containing a uniquely-named inner package for importable logic
(e.g. `models/<Name>/<name_pkg>/`) plus thin `train.py` / `evaluate.py`
entrypoints. Then:

1. Add `models/<Name>` to `extraPaths` in `[tool.pyright]` and to the list in
   `tests/models/conftest.py` (so its inner package resolves for pyright/pytest).
2. **Eval contract:** `evaluate.py` runs inference, builds a `GenerativeEvalInputs`,
   fills a complete `ExperimentMetadata` (`model_name`, `task_type`, all
   `hyperparameters`, `architecture`, `dataset_id`, `dataset_hash`), and calls
   `epp_core.runner.evaluate_model(...)`. Models never compute metrics or build
   reports themselves.
3. Add a fast pure-logic test and a `@pytest.mark.slow` end-to-end smoke test.

## Testing

- `just check` must pass before every commit.
- New metrics need a unit test on known inputs; new models need a fast pure-logic
  test plus a CPU tiny-config smoke test marked `@pytest.mark.slow`
  (`@pytest.mark.network` if it downloads a checkpoint). Slow tests are excluded
  from `just check` and run via `just test-slow`.
- Tests use synthetic/tiny inputs; the shared metrics are tested without any model.

## Commit / push workflow

After each completed, **test-passing** change: `git add -A && git commit && git
push` to `main` (unless on a feature branch — then push the branch). Interpreted
as *per logical working change*, not every file edit; never commit broken states.
`just save "<msg>"` runs `just check` then add/commit/push. A `pre-commit` hook
(`.pre-commit-config.yaml`, enable with `uv run pre-commit install`) runs the same
gate. Remote: `origin` (github.com/cg-asparagine/enzyme-product-pred).

## Conventions

- ruff (line length 100; import sort + pyupgrade) and pyright (standard) — both
  via `just`. Use dataclasses for configs/metadata.
- SMILES always go through `epp_core.chem.smiles` (single canonicalization
  source); reaction strings through `epp_core.chem.reactions`.
- Don't commit weights, `experiments/`, or `data/*/processed/` (git-ignored). The
  raw EnzymeMap CSV (386 MB) exceeds GitHub's limit and is git-ignored under
  `data/EnzymeMap/raw/`; its provenance + SHA-256 live in the dataset README and
  build manifest.
- Pandas/RDKit/transformers are loosely typed; prefer a local `cast(...)` over
  scattered `# type: ignore` when pyright over-narrows their return types.

## Roadmap (not yet built)

- **Enzyme sequences:** condition on amino-acid sequence (fetch UniProt via
  `protein_refs`, then sequence embeddings). The schema reserves room for a
  `sequence`/`uniprot_id` column; today we condition on EC number + organism text.
- **Smarter splits:** a sequence-similarity **enzyme-cluster split** is built —
  `enzyme_split` column, clustering in `epp_core.data.cluster` (k-mer Jaccard),
  for honest *new-enzyme* generalization tests; `just data-report <Dataset>`
  visualizes it. Still TODO: scaffold / compound-similarity splits for
  *new-chemistry* tests, alongside the v1 reaction `split`.
- **Models:** first a seq2seq reaction model (e.g. ReactionT5 / MolT5), then
  richer conditioning.
