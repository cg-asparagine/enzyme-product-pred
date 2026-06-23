"""Generative reaction-product evaluator."""

from __future__ import annotations

from pathlib import Path

from epp_core.eval.base import Evaluator, register_evaluator
from epp_core.eval.metrics.generative import (
    coverage_at_k,
    exact_set_match,
    f1_at_k,
    novelty,
    per_molecule_validity,
    precision_at_k,
    tanimoto_to_reference_distribution,
    top_k_accuracy,
    uniqueness,
    validity,
)
from epp_core.eval.types import EvalArtifacts, GenerativeEvalInputs, MetricResult, TaskType
from epp_core.plots import save_figure
from epp_core.plots.generative import (
    plot_coverage_curve,
    plot_tanimoto_hist,
    plot_top_k_accuracy,
)


@register_evaluator
class GenerativeEvaluator(Evaluator):
    task_type = TaskType.GENERATIVE
    ks: tuple[int, ...] = (1, 5, 10)

    def compute(self, inputs: GenerativeEvalInputs, run_dir: Path) -> EvalArtifacts:
        run_dir = Path(run_dir)
        refs, preds = inputs.references, inputs.predictions
        raw = inputs.raw_predictions or preds

        metrics: list[MetricResult] = []
        # Per-k metrics, presented as a matrix (rows = metric, columns = k) in the
        # top_k_metrics table below rather than as flat scalars. ``accuracy`` is the
        # hit rate (>=1 true product in the top-k); ``exact set match`` requires a
        # prediction equal to the full product set; ``precision`` is the fraction
        # of predicted products that are correct; ``sensitivity`` is recall (the
        # fraction of a reaction's true products recovered).
        accuracies = [top_k_accuracy(refs, preds, k) for k in self.ks]
        exact = [exact_set_match(refs, preds, k) for k in self.ks]
        precisions = [precision_at_k(refs, preds, k) for k in self.ks]
        coverages = [coverage_at_k(refs, preds, k) for k in self.ks]
        f1s = [f1_at_k(refs, preds, k) for k in self.ks]

        def _row(label: str, values: list[float]) -> list[str]:
            return [label, *(f"{v:.4f}" for v in values)]

        top_k_metrics = [
            ["Metric", *(f"Top-{k}" for k in self.ks)],
            _row("Accuracy (>=1 correct product)", accuracies),
            _row("Exact set match", exact),
            _row("Precision", precisions),
            _row("Sensitivity (recall)", coverages),
            _row("F1", f1s),
        ]

        metrics.append(MetricResult("validity", validity(raw)))
        metrics.append(MetricResult("per_molecule_validity", per_molecule_validity(raw)))
        metrics.append(MetricResult("uniqueness", uniqueness(raw)))
        if inputs.train_smiles is not None:
            metrics.append(MetricResult("novelty", novelty(preds, inputs.train_smiles)))

        sims = tanimoto_to_reference_distribution(refs, preds)
        if sims:
            metrics.append(MetricResult("mean_tanimoto_top1", sum(sims) / len(sims)))

        plots_dir = run_dir / "plots"
        plot_paths = [
            save_figure(plot_top_k_accuracy(self.ks, accuracies), plots_dir / "top_k_accuracy.png"),
            save_figure(plot_coverage_curve(self.ks, coverages), plots_dir / "coverage_curve.png"),
        ]
        if sims:
            plot_paths.append(
                save_figure(plot_tanimoto_hist(sims), plots_dir / "tanimoto_hist.png")
            )
        return EvalArtifacts(
            metrics=metrics, plot_paths=plot_paths, tables={"top_k_metrics": top_k_metrics}
        )
