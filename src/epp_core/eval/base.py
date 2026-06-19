"""Evaluator interface + registry.

The runner dispatches purely on :class:`~epp_core.eval.types.TaskType`, so adding
a task type means writing a new ``Evaluator`` subclass and decorating it with
``@register_evaluator`` — no changes to the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from epp_core.eval.types import EvalArtifacts, TaskType


class Evaluator(ABC):
    """Computes metrics, plots, and tables for one task type."""

    task_type: TaskType

    @abstractmethod
    def compute(self, inputs, run_dir: Path) -> EvalArtifacts:
        """Compute metrics and render plots (PNGs written under ``run_dir/plots``)."""


_REGISTRY: dict[TaskType, type[Evaluator]] = {}


def register_evaluator(cls: type[Evaluator]) -> type[Evaluator]:
    _REGISTRY[cls.task_type] = cls
    return cls


def get_evaluator(task_type: TaskType) -> Evaluator:
    if task_type not in _REGISTRY:
        raise KeyError(
            f"No evaluator registered for task type {task_type!r}. "
            f"Registered: {sorted(t.value for t in _REGISTRY)}"
        )
    return _REGISTRY[task_type]()
