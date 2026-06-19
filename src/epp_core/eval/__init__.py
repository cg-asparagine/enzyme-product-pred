"""Task-aware evaluation framework.

Importing this package registers the built-in evaluators (for their
``@register_evaluator`` side effect) so :func:`get_evaluator` can dispatch on
task type. Only the generative task is implemented today; classification /
multi-label evaluators can be added later behind the same registry.
"""

from epp_core.eval import generative as _generative  # noqa: F401  (registration side effect)
from epp_core.eval.base import Evaluator, get_evaluator, register_evaluator
from epp_core.eval.types import (
    EvalArtifacts,
    GenerativeEvalInputs,
    MetricResult,
    TaskType,
)

__all__ = [
    "EvalArtifacts",
    "Evaluator",
    "GenerativeEvalInputs",
    "MetricResult",
    "TaskType",
    "get_evaluator",
    "register_evaluator",
]
