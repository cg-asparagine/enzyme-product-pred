# Justfile — task runner for enzyme-product-pred.
# Requires `just` (https://just.systems) and `uv` (https://docs.astral.sh/uv/).
# List recipes with `just --list`.

# Sync the virtual environment from pyproject.toml / uv.lock
install:
    uv sync

# Auto-format the codebase
format:
    uv run ruff format .

# Lint
lint:
    uv run ruff check .

# Lint and auto-fix
lint-fix:
    uv run ruff check --fix .

# Static type-check
typecheck:
    uv run pyright

# Run the fast test suite (excludes @slow tests)
test:
    uv run pytest

# Run the slow tests (real training / network downloads)
test-slow:
    uv run pytest -m slow

# Full gate run before committing: format, lint, typecheck, fast tests
check: format lint typecheck test

# Build a dataset's processed files, e.g. `just data-prep EnzymeMap`
data-prep dataset:
    uv run python data/{{dataset}}/prepare.py

# Train a model, e.g. `just train <Model>` (available once a model exists)
train model *args:
    uv run python models/{{model}}/train.py {{args}}

# Evaluate a model -> experiments/<run_id>/report.pdf
evaluate model *args:
    uv run python models/{{model}}/evaluate.py {{args}}

# Re-render a run's PDF report from its saved JSON
report run_dir:
    uv run python -m epp_core.report.generator --run {{run_dir}}

# Render a dataset / split EDA report PDF, e.g. `just data-report EnzymeMap_with_seq`
data-report dataset:
    uv run python -m epp_core.report.dataset --dataset data/{{dataset}}/processed

# Run `just check`, then add + commit + push
save msg: check
    git add -A
    git commit -m "{{msg}}"
    git push
