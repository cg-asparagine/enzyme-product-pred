# Models

One folder per model / framework. Each is a **script directory** (not an
installed package — hyphens are allowed in the folder name) that imports the
shared `epp_core` library. **No models exist yet — the first one lands next.**

Layout convention (see [CLAUDE.md](../CLAUDE.md) for the full contract):

```
models/<Name>/
  <name_pkg>/        # uniquely-named inner package (snake_case) for importable logic
    __init__.py
    config.py        # a dataclass config + a tiny SMOKE_CONFIG for CPU smoke tests
    pipeline.py      # train / inference / evaluate-run logic
  train.py           # thin entrypoint: parse args -> call pipeline   (`just train <Name>`)
  evaluate.py        # thin entrypoint: inference -> evaluate_model    (`just evaluate <Name>`)
  README.md          # task framing, run commands, headline metrics
```

**Eval contract:** `evaluate.py` runs inference, builds a `GenerativeEvalInputs`
and a complete `ExperimentMetadata`, then calls
`epp_core.runner.evaluate_model(...)`, which writes
`experiments/<run_id>/{metadata.json, metrics.json, plots/, report.pdf}`. Models
never compute metrics or build reports themselves.

**Register a new model** for tooling: add `models/<Name>` to `extraPaths` in
`[tool.pyright]` (`pyproject.toml`) and to `tests/models/conftest.py`, then add a
fast pure-logic test plus a `@pytest.mark.slow` end-to-end smoke test.
