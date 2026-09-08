"""Evaluator implementations and the registry that names them.

:func:`default_registry` is the process-wide registry holding the built-in
evaluator types. It is built lazily and cached, so importing this package costs
nothing until an evaluator is actually needed.

Third-party evaluators are deliberately absent from it. They live in
:mod:`llm_eval_lab.evaluators.plugins`, which this module does not import: an
opt-in feature that is off by default must not be reachable simply because
something touched the evaluator package.
"""

from functools import lru_cache

from llm_eval_lab.evaluators.base import BaseEvaluator, NoParams, apply_on_error
from llm_eval_lab.evaluators.deterministic import DETERMINISTIC_EVALUATORS
from llm_eval_lab.evaluators.json_evaluators import JSON_EVALUATORS
from llm_eval_lab.evaluators.judge.evaluator import JUDGE_EVALUATORS
from llm_eval_lab.evaluators.registry import EvaluatorInfo, EvaluatorRegistry
from llm_eval_lab.evaluators.similarity import SIMILARITY_EVALUATORS

BUILTIN_EVALUATORS = (
    *DETERMINISTIC_EVALUATORS,
    *JSON_EVALUATORS,
    *SIMILARITY_EVALUATORS,
    *JUDGE_EVALUATORS,
)
"""Every evaluator type shipped with the project, in registration order."""


@lru_cache(maxsize=1)
def default_registry() -> EvaluatorRegistry:
    """Return the process-wide registry of built-in evaluator types."""
    registry = EvaluatorRegistry()
    registry.register_all(BUILTIN_EVALUATORS)
    return registry


__all__ = [
    "BUILTIN_EVALUATORS",
    "DETERMINISTIC_EVALUATORS",
    "JSON_EVALUATORS",
    "JUDGE_EVALUATORS",
    "SIMILARITY_EVALUATORS",
    "BaseEvaluator",
    "EvaluatorInfo",
    "EvaluatorRegistry",
    "NoParams",
    "apply_on_error",
    "default_registry",
]
