"""The two structural evaluators: `json_valid` and `json_schema`.

Both answer a question about the SHAPE of a response rather than its content,
and both are fully deterministic: no model call, no network, no randomness.

**Why a malformed response is FAILED and not ERROR.** A model asked for JSON
that returns prose has answered wrongly; the evaluator worked perfectly and
reached a conclusion. ERROR is reserved for the evaluator itself being unable to
decide, which here means only a schema this module cannot use - and that is
caught at benchmark LOAD time, so it never reaches a run.

**Why remote ``$ref`` is refused.** A benchmark file is shared data. A schema
carrying ``{"$ref": "https://example.invalid/schema.json"}`` would make loading
somebody's benchmark issue an outbound request to a host that benchmark chose,
during validation, before anything was scored. There is no version of this
project in which that is acceptable, so a ``$ref`` that is not a local pointer
(``#/...``) is a benchmark validation error naming the case and the field.
Local pointers keep working, which is what real schemas actually use.

**Why a benchmark-supplied `pattern` runs in a separate process.** JSON Schema's
``pattern`` and ``patternProperties`` keywords are regular expressions written by
whoever wrote the benchmark file, matched against text written by the model. That
is exactly the hazard the ``regex`` evaluator already guards against, arriving
through a different door: ``{"pattern": "^(a+)+$"}`` is a dozen characters, and
against a 26-character non-matching string it takes seconds, doubling per added
character. Run on the event loop it would not merely fail one case - a CPython
regex match holds the GIL for its whole duration, so it freezes the loop, stalls
every concurrent case, and takes an API-hosted run's HTTP server down with it.

So a schema that asserts a pattern is validated in a SEPARATE PROCESS under a
wall-clock bound, through the same :mod:`llm_eval_lab.evaluators.bounded` worker
the ``regex`` evaluator uses; on expiry the process is killed and the evaluation
is ``status=ERROR`` with ``metadata["error_code"] = "schema_timeout"``. A schema
with no pattern keyword anywhere in it - the overwhelming majority - is validated
in process, because there is no untrusted regex to bound and an interpreter start
per case would be a real cost for no benefit. Which path a schema takes is
decided once, at load time, by the same walk that refuses remote references.

Format assertions stay off, which is jsonschema's default: no ``format_checker``
is ever passed, so a ``format`` keyword is an annotation and its regexes never
run. That is asserted by a test rather than left as a reading of the library.

**Partial-credit scoring for `json_schema`.** ``score = 1 / (n + 1)`` where ``n``
is the number of distinct schema violations, which is the ``1 - n/(n+1)`` the
registry table states. It is monotonically decreasing, never negative, never
zero, and treats the first violation as the expensive one: a response with one
error scores 0.5 and a response with three scores 0.25. Every violation is
listed in ``metadata["errors"]``, so the number is never the only evidence.
"""

import builtins
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, ClassVar

# `jsonschema` ships no type stubs and `types-jsonschema` is not in the dev
# dependency group, which lives in the frozen `pyproject.toml`. Ignored here,
# narrowly and with a reason, rather than loosening the strict mypy settings.
import jsonschema  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.bounded import WorkerError, run_worker
from llm_eval_lab.models import (
    EvaluationContext,
    EvaluationResult,
    JSONValue,
    ModelResponse,
    ResolvedCase,
)
from llm_eval_lab.utils.json import canonical_json_bytes
from llm_eval_lab.utils.time import monotonic_ms

MAX_SCHEMA_ERRORS = 50
"""Cap on how many violations one result lists.

A schema with a large ``anyOf`` can produce thousands of messages for one
response. The count that feeds the score is capped here too, so the score stays
comparable across responses and the stored metadata stays a readable list rather
than a wall of text. ``metadata["errors_truncated"]`` says when the cap bit.
"""

MAX_ERROR_MESSAGE_CHARS = 300
"""Cap on one violation message; a validator can quote an entire instance."""

MAX_SCHEMA_DEPTH = 64
"""How deeply a schema may nest before it is refused at benchmark load time.

Generous for a schema a person wrote, and far below CPython's recursion limit.
Without it, a schema nested past roughly 110 levels raised ``RecursionError`` out
of the load-time walk - and the registry's validation pass catches
``pydantic.ValidationError`` only, so it escaped as a traceback from
``llm-eval validate`` and as an opaque 500 from the validate endpoint, rather
than as a field error naming the case.
"""

MAX_SCHEMA_BYTES = 65_536
"""Cap on one schema's canonical JSON size, applied at benchmark load time.

The suite file as a whole is already capped, but one 8 MiB schema inside a valid
suite is not a suite-size problem: it is handed to a validator, walked per case,
and shipped to a worker process per case.
"""

MAX_SCHEMA_INSTANCE_CHARS = 1_000_000
"""Cap on the model output handed to a schema WORKER, so the pipe payload is bounded.

Oversized input is an ERROR rather than a truncation: half a JSON document is not
a JSON document, and validating a prefix would report violations of a shape the
model never produced.
"""

DEFAULT_SCHEMA_VALIDATION_TIMEOUT_S = 2.0
"""Default bound on validating one response against a pattern-bearing schema."""

SCHEMA_WORKER_STARTUP_ALLOWANCE_S = 1.0
"""Added to the validation budget to cover interpreter start and the jsonschema import.

Measured at well under 200 ms. It is a separate allowance rather than part of the
budget so the configured number means what it says - a bound on the VALIDATION -
instead of silently shrinking as the import gets slower.
"""

SCHEMA_REGEX_KEYWORDS: frozenset[str] = frozenset({"pattern", "patternProperties"})
"""Schema keywords whose values are regular expressions compiled from benchmark data.

``format`` is deliberately absent. Format assertions require a ``format_checker``
this evaluator never passes, so a ``format`` keyword compiles nothing; a test
pins that. Adding it here would send every schema using ``format: date-time`` -
which is most of them - through a subprocess for no reason.
"""

SCHEMA_WORKER_SOURCE = """\
import json, sys

import jsonschema

request = json.loads(sys.stdin.read())
limit = request["max_errors"]
try:
    schema = request["schema"]
    instance = json.loads(request["instance_text"])
    validator = jsonschema.validators.validator_for(schema)(schema)
    found = []
    for index, error in enumerate(validator.iter_errors(instance)):
        if index > limit:
            break
        found.append(
            {
                "path": [str(part) for part in error.absolute_path],
                "message": error.message,
            }
        )
except Exception as exc:
    json.dump({"error": type(exc).__name__, "message": str(exc)}, sys.stdout)
else:
    json.dump({"errors": found}, sys.stdout)
"""
"""The whole schema worker program.

Started with ``-I`` but NOT ``-S``: it needs ``jsonschema`` from site-packages,
which ``-S`` would hide. It still reads no environment, imports nothing from this
project, and shares no memory with the run. It stops collecting one past the
cap, so a schema with a wide ``anyOf`` cannot make it materialise a million
error objects.
"""

_FENCE = re.compile(
    r"\A\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?[ \t]*```\s*\Z",
    re.DOTALL,
)
"""Matches a response that is exactly ONE markdown fence and nothing else.

Anchored at both ends and with a single lazy body group, so there is no nested
quantifier and no backtracking blowup: this pattern is a constant of this
module, never author-supplied, and it is applied to model output that can be
large. One fence, deliberately - a response containing two fenced blocks is
ambiguous about which one is the answer, and guessing is how an evaluator starts
inventing results.
"""

_LOCAL_REF_PREFIX = "#"


def strip_markdown_fence(text: str) -> tuple[str, bool]:
    """Return `text` with a single wrapping markdown fence removed.

    Returns the possibly-unwrapped text and whether a fence was actually
    removed, so a caller can record that the model wrapped its answer rather
    than silently normalizing the difference away.
    """
    match = _FENCE.match(text)
    if match is None:
        return text, False
    return match.group("body"), True


def unwrap_json_payload(text: str, *, allow_markdown_fence: bool) -> tuple[str, bool]:
    """Return the text that will be parsed, and whether a fence was removed.

    Separate from parsing so a caller can report the fence fact even when the
    parse then fails. Reporting ``fenced: False`` for a fenced block whose body
    was bad JSON described the response wrongly, in exactly the case somebody is
    most likely to go and read the metadata.
    """
    if allow_markdown_fence:
        return strip_markdown_fence(text)
    return text, False


def parse_json_payload(text: str, *, allow_markdown_fence: bool) -> tuple[JSONValue, str, bool]:
    """Parse the JSON a response carries.

    Returns the parsed value, the exact text that was parsed, and whether a
    markdown fence had to be removed first.

    Raises:
        json.JSONDecodeError: when the text is not JSON. Callers turn this into
            a FAILED evaluation, never an ERROR: the model answered wrongly.
    """
    payload, fenced = unwrap_json_payload(text, allow_markdown_fence=allow_markdown_fence)
    return json.loads(payload), payload, fenced


@dataclass
class SchemaScan:
    """What one bounded walk over a schema found."""

    remote_ref: tuple[str, str] | None = None
    """Dotted location and value of the first non-local ``$ref``, if any."""

    uses_regex: bool = False
    """Whether any keyword in the schema compiles a benchmark-supplied regex."""


def _scan_schema(schema: JSONValue, path: str, scan: SchemaScan, depth: int) -> None:
    """Walk a schema once, under an explicit depth bound.

    One walk rather than three, because each of the three questions it answers -
    is there a remote reference, does anything here compile a regex, and is this
    thing absurdly deep - has to be answered before the schema is used, and a
    hostile schema should be walked once rather than once per question.

    Raises:
        ValueError: when the schema nests past :data:`MAX_SCHEMA_DEPTH`. Raised
            from a pydantic validator, so it arrives as a field error naming the
            case rather than as a ``RecursionError`` traceback.
    """
    if depth > MAX_SCHEMA_DEPTH:
        msg = (
            f"{path} nests deeper than {MAX_SCHEMA_DEPTH} levels; a schema this deep is "
            f"far more likely to be a denial-of-service attempt than a description of data"
        )
        raise ValueError(msg)

    if isinstance(schema, dict):
        for key, value in schema.items():
            here = f"{path}.{key}"
            if key == "$ref" and isinstance(value, str) and not value.startswith(_LOCAL_REF_PREFIX):
                scan.remote_ref = scan.remote_ref or (here, value)
            if key in SCHEMA_REGEX_KEYWORDS:
                scan.uses_regex = True
            _scan_schema(value, here, scan, depth + 1)
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            _scan_schema(item, f"{path}[{index}]", scan, depth + 1)


def scan_schema(schema: JSONValue) -> SchemaScan:
    """Return what a bounded walk over `schema` found.

    Raises:
        ValueError: when the schema nests past :data:`MAX_SCHEMA_DEPTH`.
    """
    scan = SchemaScan()
    _scan_schema(schema, "schema", scan, 0)
    return scan


def _render_violation(path_parts: Sequence[str], message: str) -> str:
    """Render one schema violation as a short, located, single-line message."""
    location = "/".join(path_parts)
    flat = message.replace("\n", " ")
    if len(flat) > MAX_ERROR_MESSAGE_CHARS:
        flat = flat[: MAX_ERROR_MESSAGE_CHARS - 3] + "..."
    return f"{location or '<root>'}: {flat}"


def _sorted_violations(found: Sequence[tuple[list[str], str]]) -> list[str]:
    """Order violations by their location, then render them.

    Path parts are compared AS STRINGS. An instance mixing array indices with
    property names yields paths holding both ints and strs, and comparing those
    directly raises `TypeError` - which would turn a schema violation into an
    evaluator crash.
    """
    return [
        _render_violation(parts, message)
        for parts, message in sorted(found, key=lambda item: item[0])
    ]


class JsonValidParams(BaseModel):
    """Parameters for the `json_valid` evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    require_object: bool = Field(
        default=False,
        description="Require the top-level value to be a JSON object rather than any JSON value.",
    )
    allow_markdown_fence: bool = Field(
        default=True,
        description="Strip a single wrapping ```-fence before parsing.",
    )


class JsonValidEvaluator(BaseEvaluator[JsonValidParams]):
    """Whether the response parses as JSON, optionally as a JSON object."""

    type: ClassVar[str] = "json_valid"
    params_model: ClassVar[builtins.type[BaseModel]] = JsonValidParams
    needs_expected: ClassVar[bool] = False

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Parse the response and report whether it is usable JSON."""
        del case, context
        started = monotonic_ms()
        if response.output_text is None:
            return self.error(
                kind="runtime",
                message="response carries no output text",
                error_code="no_output",
                duration_ms=monotonic_ms() - started,
            )

        payload, fenced = unwrap_json_payload(
            response.output_text,
            allow_markdown_fence=self.params.allow_markdown_fence,
        )
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            return self.verdict(
                passed=False,
                explanation=f"response is not valid JSON: {exc.msg} at line {exc.lineno}",
                metadata={"fenced": fenced, "parse_error": exc.msg},
                duration_ms=monotonic_ms() - started,
            )

        if self.params.require_object and not isinstance(value, dict):
            return self.verdict(
                passed=False,
                explanation=(
                    f"response parsed as JSON but the top-level value is "
                    f"{type(value).__name__}, not an object"
                ),
                metadata={"fenced": fenced, "top_level_type": type(value).__name__},
                duration_ms=monotonic_ms() - started,
            )

        return self.verdict(
            passed=True,
            metadata={"fenced": fenced, "top_level_type": type(value).__name__},
            duration_ms=monotonic_ms() - started,
        )


class JsonSchemaParams(BaseModel):
    """Parameters for the `json_schema` evaluator.

    ``schema`` is checked against its own metaschema by a validator, so an
    unusable schema is a benchmark validation error naming the case and the
    field rather than an error on every case at run time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # `json_schema` is the field name a benchmark author writes; `schema` would
    # shadow pydantic's own deprecated `BaseModel.schema` classmethod.
    json_schema: dict[str, JSONValue] = Field(
        alias="schema",
        description="The JSON Schema the response must satisfy. Remote $ref is refused.",
    )
    allow_markdown_fence: bool = Field(
        default=True,
        description="Strip a single wrapping ```-fence before parsing.",
    )

    validation_timeout_s: float = Field(
        default=DEFAULT_SCHEMA_VALIDATION_TIMEOUT_S,
        gt=0.0,
        le=60.0,
        description=(
            "Wall-clock bound on validating one response, applied only when the schema "
            "asserts a pattern. Further capped by settings.max_regex_match_s."
        ),
    )

    @field_validator("json_schema")
    @classmethod
    def _schema_is_usable(cls, value: dict[str, JSONValue]) -> dict[str, JSONValue]:
        """Reject a schema this evaluator cannot use, at benchmark load time.

        Four refusals, all of them here so an unusable schema is a benchmark
        validation error naming the case and the field rather than an identical
        failure on every case at run time: a schema too large to be worth
        shipping to a worker per case, one nested deeply enough to be a denial of
        service rather than a description, one carrying a non-local ``$ref``
        (which would make validation fetch a URL the benchmark chose), and one
        its own metaschema rejects.
        """
        try:
            size = len(canonical_json_bytes(value))
        except (TypeError, ValueError) as exc:
            msg = f"schema is not representable as JSON: {exc}"
            raise ValueError(msg) from exc
        if size > MAX_SCHEMA_BYTES:
            msg = (
                f"schema is {size} bytes, over the {MAX_SCHEMA_BYTES}-byte limit; it is "
                f"walked and validated once per case, so an oversized one is a cost paid "
                f"on every response"
            )
            raise ValueError(msg)

        scan = scan_schema(value)
        if scan.remote_ref is not None:
            location, reference = scan.remote_ref
            msg = (
                f"{location} is a non-local reference to {reference!r}; a benchmark file "
                f"must not make validation fetch a remote document. Inline the referenced "
                f"schema and use a local '#/...' pointer instead."
            )
            raise ValueError(msg)
        try:
            jsonschema.validators.validator_for(value).check_schema(value)
        except SchemaError as exc:
            msg = f"invalid JSON Schema: {exc.message}"
            raise ValueError(msg) from exc
        return value

    def asserts_a_pattern(self) -> bool:
        """Report whether this schema compiles a benchmark-supplied regex."""
        return scan_schema(self.json_schema).uses_regex

    def validator(self) -> Any:
        """Return a jsonschema validator bound to this schema.

        No ``format_checker`` is passed, so ``format`` is an annotation rather
        than an assertion: whether a benchmark scores the same on two machines
        must not depend on which optional format libraries happen to be
        installed there.
        """
        return jsonschema.validators.validator_for(self.json_schema)(self.json_schema)


class JsonSchemaEvaluator(BaseEvaluator[JsonSchemaParams]):
    """Whether the response satisfies a JSON Schema, with partial credit.

    Format assertions are deliberately off, which is jsonschema's default: a
    keyword such as ``format: email`` is an annotation in the specification, and
    turning it into an assertion here would make the same benchmark score
    differently depending on which optional format libraries happen to be
    installed.
    """

    type: ClassVar[str] = "json_schema"
    params_model: ClassVar[builtins.type[BaseModel]] = JsonSchemaParams
    needs_expected: ClassVar[bool] = False

    def __init__(self, spec: Any, params: JsonSchemaParams) -> None:
        """Compile the schema once, and decide once which path it takes."""
        super().__init__(spec, params)
        self._validator = params.validator()
        self._needs_worker = params.asserts_a_pattern()

    async def _violations(
        self,
        payload: str,
        value: JSONValue,
        budget: float,
    ) -> tuple[list[str], bool]:
        """Return the schema violations, capped, and whether the cap bit.

        Takes the in-process path unless the schema asserts a pattern, in which
        case the whole validation runs in a killable subprocess. Both paths stop
        collecting one past the cap: `sorted()` over the full generator was what
        let a wide ``anyOf`` against a long array materialise hundreds of
        megabytes of error objects before anything was truncated.

        Raises:
            TimeoutError: when a pattern-bearing schema exceeded `budget`.
            WorkerError: when the worker process could not run at all.
        """
        if not self._needs_worker:
            found = [
                ([str(part) for part in error.absolute_path], error.message)
                for error in islice(self._validator.iter_errors(value), MAX_SCHEMA_ERRORS + 1)
            ]
        else:
            request = json.dumps(
                {
                    "schema": self.params.json_schema,
                    "instance_text": payload,
                    "max_errors": MAX_SCHEMA_ERRORS,
                }
            ).encode("utf-8")
            stdout = await run_worker(
                SCHEMA_WORKER_SOURCE,
                request,
                timeout_s=budget + SCHEMA_WORKER_STARTUP_ALLOWANCE_S,
                stdlib_only=False,
            )
            result = json.loads(stdout.decode("utf-8"))
            if "error" in result:
                msg = f"{result['error']}: {result['message']}"
                raise WorkerError(msg)
            found = [(entry["path"], entry["message"]) for entry in result["errors"]]

        truncated = len(found) > MAX_SCHEMA_ERRORS
        return _sorted_violations(found[:MAX_SCHEMA_ERRORS]), truncated

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Validate the response against the configured schema."""
        del case
        started = monotonic_ms()
        if response.output_text is None:
            return self.error(
                kind="runtime",
                message="response carries no output text",
                error_code="no_output",
                duration_ms=monotonic_ms() - started,
            )

        payload, fenced = unwrap_json_payload(
            response.output_text,
            allow_markdown_fence=self.params.allow_markdown_fence,
        )
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            # `raw_score` means "number of schema violations" on every other path
            # through this evaluator, so leaving it unset for the worst outcome
            # made a consumer reading it see None exactly when the response was
            # least usable. A response that is not JSON violates the schema once,
            # at the root, and that is what is reported.
            message = f"response is not valid JSON: {exc.msg} at line {exc.lineno}"
            return self.verdict(
                passed=False,
                score=0.0,
                raw_score=1.0,
                explanation=message,
                metadata={
                    "fenced": fenced,
                    "parse_error": exc.msg,
                    "n_errors": 1,
                    "errors": [f"<root>: {message}"],
                    "errors_truncated": False,
                },
                duration_ms=monotonic_ms() - started,
            )

        budget = min(self.params.validation_timeout_s, context.settings.max_regex_match_s)
        if self._needs_worker and len(payload) > MAX_SCHEMA_INSTANCE_CHARS:
            return self.error(
                kind="runtime",
                message=(
                    f"response is {len(payload)} characters, over the "
                    f"{MAX_SCHEMA_INSTANCE_CHARS}-character limit for validating against a "
                    f"schema that asserts a pattern"
                ),
                error_code="schema_instance_too_large",
                duration_ms=monotonic_ms() - started,
            )
        try:
            violations, truncated = await self._violations(payload, value, budget)
        except TimeoutError:
            return self.error(
                kind="timeout",
                message=(
                    f"validating the response against this schema exceeded {budget}s and was "
                    f"abandoned; a 'pattern' in the schema is the usual cause"
                ),
                error_code="schema_timeout",
                metadata={"timeout_s": budget, "bounded_in_subprocess": True},
                duration_ms=monotonic_ms() - started,
            )
        except (WorkerError, RecursionError, MemoryError) as exc:
            return self.error(
                kind="runtime",
                message=f"schema validation failed: {exc}",
                error_code="schema_error",
                exception_type=type(exc).__name__,
                duration_ms=monotonic_ms() - started,
            )
        n_errors = len(violations)
        messages: JSONValue = list(violations)

        return self.verdict(
            passed=n_errors == 0,
            score=1.0 / (n_errors + 1),
            raw_score=float(n_errors),
            explanation=(
                None
                if n_errors == 0
                else f"response violates the schema in {n_errors} place(s): {violations[0]}"
            ),
            metadata={
                "fenced": fenced,
                "n_errors": n_errors,
                "errors": messages,
                "errors_truncated": truncated,
                "bounded_in_subprocess": self._needs_worker,
            },
            duration_ms=monotonic_ms() - started,
        )


JSON_EVALUATORS: tuple[type[BaseEvaluator[Any]], ...] = (
    JsonValidEvaluator,
    JsonSchemaEvaluator,
)
"""Every evaluator this module defines, in registration order."""

__all__ = [
    "JSON_EVALUATORS",
    "MAX_SCHEMA_BYTES",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_ERRORS",
    "MAX_SCHEMA_INSTANCE_CHARS",
    "SCHEMA_REGEX_KEYWORDS",
    "JsonSchemaEvaluator",
    "JsonSchemaParams",
    "JsonValidEvaluator",
    "JsonValidParams",
    "SchemaScan",
    "parse_json_payload",
    "scan_schema",
    "strip_markdown_fence",
    "unwrap_json_payload",
]
