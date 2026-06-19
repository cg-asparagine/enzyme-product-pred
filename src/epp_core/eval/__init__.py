"""Task-aware evaluation framework.

Currently only the generative task is implemented. Built-in evaluators are
registered here by importing their modules for the ``@register_evaluator`` side
effect once they exist (see :mod:`epp_core.eval.generative`); for now this
package just re-exports the shared types.
"""

from epp_core.eval.types import (
    EvalArtifacts,
    GenerativeEvalInputs,
    MetricResult,
    TaskType,
)

__all__ = [
    "EvalArtifacts",
    "GenerativeEvalInputs",
    "MetricResult",
    "TaskType",
]
