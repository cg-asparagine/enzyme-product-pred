"""Generative reaction-product evaluator."""

from __future__ import annotations

from pathlib import Path

from epp_core.eval.base import Evaluator, register_evaluator
from epp_core.eval.metrics.generative import (
    coverage_at_k,
    exact_set_match,
    novelty,
    per_molecule_validity,
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
        accuracies = [top_k_accuracy(refs, preds, k) for k in self.ks]
        exact = [exact_set_match(refs, preds, k) for k in self.ks]
        coverages = [coverage_at_k(refs, preds, k) for k in self.ks]
        for k, acc in zip(self.ks, accuracies, strict=True):
            metrics.append(MetricResult(f"top_{k}_accuracy", acc))
        for k, em in zip(self.ks, exact, strict=True):
            metrics.append(MetricResult(f"exact_set_match_top_{k}", em))
        for k, cov in zip(self.ks, coverages, strict=True):
            metrics.append(MetricResult(f"coverage_at_{k}", cov))
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
        return EvalArtifacts(metrics=metrics, plot_paths=plot_paths)
