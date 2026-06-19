"""Core types shared across the evaluation framework.

These are plain data containers with no model or heavy-library dependencies, so
they can be imported and unit-tested cheaply. Models hand the framework one of
the ``*EvalInputs`` objects; evaluators return :class:`EvalArtifacts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskType(StrEnum):
    """The kind of prediction task a model performs.

    Determines which metrics, plots, and report sections the framework produces.
    A ``StrEnum``, so ``str(TaskType.GENERATIVE) == "generative"`` and it
    serializes straight to JSON. Only ``GENERATIVE`` is implemented today;
    ``CLASSIFICATION`` / ``MULTILABEL`` are reserved for future task framings.
    """

    GENERATIVE = "generative"
    CLASSIFICATION = "classification"
    MULTILABEL = "multilabel"


@dataclass(frozen=True)
class MetricResult:
    """A single named metric.

    ``value`` may be ``None`` for metrics represented purely as a curve/plot, in
    which case the supporting data lives in ``detail``.
    """

    name: str
    value: float | None
    higher_is_better: bool = True
    detail: dict = field(default_factory=dict)


@dataclass
class EvalArtifacts:
    """Everything an evaluator produces for one run: metrics, plots, and tables."""

    metrics: list[MetricResult] = field(default_factory=list)
    plot_paths: list[str] = field(default_factory=list)
    tables: dict[str, list[list]] = field(default_factory=dict)

    def metric(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)

    def scalar_dict(self) -> dict[str, float | None]:
        return {m.name: m.value for m in self.metrics}


@dataclass
class GenerativeEvalInputs:
    """Inputs for evaluating a generative reaction-product model.

    ``references[i]`` is the set of ground-truth product SMILES for reaction
    ``i`` (the product-side molecules). ``predictions[i]`` is the model's
    *ranked* list of predicted product-side SMILES (each entry may itself be a
    dot-joined multi-molecule product set). ``raw_predictions[i]`` is the
    optional unfiltered generation output (used for validity/uniqueness), and
    ``train_smiles`` enables novelty.
    """

    references: list[list[str]]
    predictions: list[list[str]]
    raw_predictions: list[list[str]] | None = None
    train_smiles: set[str] | None = None

    def __post_init__(self) -> None:
        if len(self.references) != len(self.predictions):
            raise ValueError(
                f"references ({len(self.references)}) and predictions "
                f"({len(self.predictions)}) must be the same length"
            )
