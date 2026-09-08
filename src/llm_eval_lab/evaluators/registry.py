"""The evaluator registry and the load-time validation pass it powers.

Registration is explicit: a class is added by name, and the name is what a
benchmark file writes. There is no import-by-dotted-path and no expression
language, so a benchmark file can never cause code to be imported or executed
(decision D-CUSTOM). Opt-in entry-point plugins arrive in a later slice and go
through this same registry.

:meth:`EvaluatorRegistry.validate_specs` is what makes an unusable evaluator a
BENCHMARK VALIDATION error rather than a runtime one: an unknown type, a
parameter the evaluator's own ``params_model`` refuses, an uncompilable regex,
all reported with the case id and the dotted field location, before a single
request is issued.
"""

import builtins
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from llm_eval_lab.datasets.loader import format_location
from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.models import (
    BenchmarkSuite,
    EvaluatorConfigError,
    EvaluatorSpec,
    FieldError,
)

MAX_INPUT_REPR_CHARS = 120
"""Matches the loader's cap, so one field error looks the same wherever it came from."""


@dataclass(frozen=True)
class EvaluatorInfo:
    """The catalog entry for one registered evaluator type."""

    type: str
    # `builtins.type` because the `type` field above shadows the builtin here.
    params_model: builtins.type[BaseModel]
    needs_expected: bool
    is_model_graded: bool
    summary: str

    def params_schema(self) -> dict[str, Any]:
        """Return the JSON Schema of this evaluator's parameter model."""
        schema: dict[str, Any] = self.params_model.model_json_schema()
        return schema


def _truncate(value: object) -> str:
    """Return a short repr of an offending parameter value."""
    text = repr(value)
    if len(text) <= MAX_INPUT_REPR_CHARS:
        return text
    return text[: MAX_INPUT_REPR_CHARS - 3] + "..."


def _summary_of(evaluator_class: type[BaseEvaluator[Any]]) -> str:
    """Return the first line of an evaluator class docstring."""
    doc = (evaluator_class.__doc__ or "").strip()
    return doc.splitlines()[0] if doc else ""


class EvaluatorRegistry:
    """Maps an evaluator type name to its class, and validates specs against it."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._classes: dict[str, type[BaseEvaluator[Any]]] = {}

    def register(self, evaluator_class: type[BaseEvaluator[Any]]) -> None:
        """Register one evaluator class under its declared `type`.

        Raises:
            EvaluatorConfigError: if the name is already taken. Silent
                replacement would let a plugin shadow a built-in evaluator and
                change what a suite means without anyone seeing it.
        """
        name = evaluator_class.type
        existing = self._classes.get(name)
        if existing is not None and existing is not evaluator_class:
            msg = (
                f"evaluator type {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}"
            )
            raise EvaluatorConfigError(msg)
        self._classes[name] = evaluator_class

    def register_all(self, evaluator_classes: Iterable[type[BaseEvaluator[Any]]]) -> None:
        """Register several evaluator classes."""
        for evaluator_class in evaluator_classes:
            self.register(evaluator_class)

    def types(self) -> tuple[str, ...]:
        """Return every registered type name, sorted."""
        return tuple(sorted(self._classes))

    def has(self, evaluator_type: str) -> bool:
        """Report whether a type name is registered."""
        return evaluator_type in self._classes

    def info(self, evaluator_type: str) -> EvaluatorInfo:
        """Return the catalog entry for one type.

        Raises:
            EvaluatorConfigError: when the type is not registered.
        """
        evaluator_class = self._classes.get(evaluator_type)
        if evaluator_class is None:
            msg = (
                f"unknown evaluator type {evaluator_type!r}; "
                f"registered types are {', '.join(self.types())}"
            )
            raise EvaluatorConfigError(msg)
        return EvaluatorInfo(
            type=evaluator_class.type,
            params_model=evaluator_class.params_model,
            needs_expected=evaluator_class.needs_expected,
            is_model_graded=evaluator_class.is_model_graded,
            summary=_summary_of(evaluator_class),
        )

    def catalog(self) -> tuple[EvaluatorInfo, ...]:
        """Return the catalog entry for every registered type."""
        return tuple(self.info(name) for name in self.types())

    def create(self, spec: EvaluatorSpec) -> BaseEvaluator[Any]:
        """Instantiate the evaluator a spec names, validating its parameters.

        Raises:
            EvaluatorConfigError: for an unknown type or unusable parameters.
                Both are already load-time errors via :meth:`validate_specs`;
                reaching this raise means the spec bypassed validation.
        """
        evaluator_class = self._classes.get(spec.type)
        if evaluator_class is None:
            msg = (
                f"unknown evaluator type {spec.type!r}; "
                f"registered types are {', '.join(self.types())}"
            )
            raise EvaluatorConfigError(msg)
        try:
            params = evaluator_class.params_model.model_validate(spec.params)
        except (ValidationError, RecursionError) as exc:
            msg = f"invalid parameters for evaluator {spec.type!r}: {exc}"
            raise EvaluatorConfigError(msg) from exc
        return evaluator_class(spec, params)

    def _spec_errors(
        self,
        spec: EvaluatorSpec,
        *,
        file: str,
        prefix: str,
        case_id: str | None,
    ) -> list[FieldError]:
        """Validate one spec, returning a field error per problem found."""
        evaluator_class = self._classes.get(spec.type)
        if evaluator_class is None:
            return [
                FieldError(
                    file=file,
                    location=f"{prefix}.type",
                    case_id=case_id,
                    message=(
                        f"unknown evaluator type {spec.type!r}; "
                        f"registered types are {', '.join(self.types())}"
                    ),
                    input_repr=_truncate(spec.type),
                )
            ]
        try:
            evaluator_class.params_model.model_validate(spec.params)
        except ValidationError as exc:
            return [
                FieldError(
                    file=file,
                    location=f"{prefix}.params.{format_location(detail['loc'])}".removesuffix(
                        ".<suite>"
                    ),
                    case_id=case_id,
                    message=detail["msg"],
                    input_repr=_truncate(detail.get("input")),
                )
                for detail in exc.errors()
            ]
        except RecursionError:
            # A parameter model may walk its own input, and a deeply nested
            # benchmark value can exhaust the stack inside that walk. It is not a
            # `ValidationError`, so without this it escaped the whole validation
            # pass: `llm-eval validate` printed a traceback and the validate
            # endpoint returned an opaque 500, in both cases for a file whose
            # problem is entirely describable. Individual parameter models bound
            # their own depth; this is the net under all of them.
            return [
                FieldError(
                    file=file,
                    location=f"{prefix}.params",
                    case_id=case_id,
                    message=(
                        "parameters nest too deeply to validate; a value this deep is far "
                        "more likely to be a denial-of-service attempt than real configuration"
                    ),
                    input_repr="<too deeply nested to render>",
                )
            ]
        return []

    def validate_specs(self, suite: BenchmarkSuite, file: str) -> Sequence[FieldError]:
        """Validate every evaluator spec in a suite, collecting all problems.

        Signature matches ``datasets.loader.SuiteValidator`` so it can be handed
        to :func:`~llm_eval_lab.datasets.loader.load_suite` directly, which is
        what lets structural and evaluator errors be reported in one pass
        without ``datasets`` importing ``evaluators``.
        """
        errors: list[FieldError] = []
        for index, spec in enumerate(suite.defaults.evaluators):
            errors.extend(
                self._spec_errors(
                    spec,
                    file=file,
                    prefix=f"defaults.evaluators[{index}]",
                    case_id=None,
                )
            )
        for case_index, case in enumerate(suite.cases):
            for index, spec in enumerate(case.evaluators):
                errors.extend(
                    self._spec_errors(
                        spec,
                        file=file,
                        prefix=f"cases[{case_index}].evaluators[{index}]",
                        case_id=case.id,
                    )
                )
        return errors
