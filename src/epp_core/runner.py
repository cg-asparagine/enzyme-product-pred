"""The evaluation orchestrator that ties metadata → metrics → plots → PDF.

A model's ``evaluate.py`` runs inference, builds the task-appropriate
``*EvalInputs`` and a complete :class:`ExperimentMetadata`, then calls
:func:`evaluate_model`. This module is deliberately model-agnostic — it never
imports torch/transformers.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from epp_core.eval import get_evaluator  # importing registers the built-in evaluators
from epp_core.io import new_run_dir, write_json
from epp_core.metadata.capture import ExperimentMetadata
from epp_core.report.generator import render_report


def evaluate_model(
    *,
    inputs,
    metadata: ExperimentMetadata,
    run_root: str | Path = "experiments",
    run_dir: str | Path | None = None,
) -> Path:
    """Evaluate ``inputs`` for ``metadata.task_type`` and write a run directory.

    Produces ``<run_dir>/{metadata.json, metrics.json, plots/*.png, report.pdf}``
    and returns the run directory path.
    """
    if run_dir is None:
        run_dir = new_run_dir(run_root, metadata.model_name, metadata.git.get("commit", ""))
    else:
        run_dir = Path(run_dir)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    metadata.run_id = run_dir.name

    evaluator = get_evaluator(metadata.task_type)
    artifacts = evaluator.compute(inputs, run_dir)

    write_json(run_dir / "metadata.json", metadata.to_dict())
    write_json(run_dir / "metrics.json", [asdict(m) for m in artifacts.metrics])
    write_json(run_dir / "tables.json", artifacts.tables)
    render_report(metadata.to_dict(), artifacts, run_dir / "report.pdf")
    return run_dir
