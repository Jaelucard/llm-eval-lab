"""Benchmark suite loading, with every error in the file reported at once.

Two properties this module exists to guarantee.

**Nothing in a benchmark file executes.** Parsing is ``yaml.safe_load`` only,
so a ``!!python/object/apply:os.system`` tag is a load error rather than a
shell command. The file is size-capped before it is parsed, so a hostile or
accidental multi-gigabyte file is refused rather than read into memory.

**Validation reports everything, not the first thing.** A pydantic
``ValidationError`` carries every problem it found; this module translates all
of them into :class:`~llm_eval_lab.models.FieldError` records that name the
file, the dotted location, the offending case id and a truncated repr of the
input. Fixing a malformed suite is then one pass instead of ten.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm_eval_lab.models import (
    BenchmarkSuite,
    BenchmarkValidationError,
    FieldError,
    JSONValue,
)
from llm_eval_lab.utils.paths import PathEscapeError, resolve_under_root

DEFAULT_MAX_SUITE_BYTES = 8_388_608
"""Fallback cap when no `Settings` value is supplied. Matches `settings.max_suite_bytes`."""

MAX_INPUT_REPR_CHARS = 120
"""How much of an offending value is echoed back. Enough to recognize, short enough to read."""

_YAML_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml"})
_JSON_SUFFIXES: frozenset[str] = frozenset({".json"})

type SuiteValidator = Callable[[BenchmarkSuite, str], Sequence[FieldError]]
"""An extra validation pass over an already-structurally-valid suite.

Evaluator-level checks (unknown type, params that fail the evaluator's own
`params_model`, an uncompilable regex) live in the `evaluators` package, which
sits ABOVE `datasets` in the layered dependency contract and therefore cannot
be imported from here. Injecting them keeps the layering intact while still
letting one `load_suite` call report structural and evaluator errors together.
"""


def _truncate(value: object) -> str:
    """Return a short, single-line repr of an offending input value."""
    text = repr(value)
    if len(text) <= MAX_INPUT_REPR_CHARS:
        return text
    return text[: MAX_INPUT_REPR_CHARS - 3] + "..."


def format_location(loc: Sequence[object]) -> str:
    """Render a pydantic error location as `cases[7].evaluators[0].params.pattern`."""
    parts: list[str] = []
    for element in loc:
        if isinstance(element, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{element}]"
            else:
                parts.append(f"[{element}]")
        else:
            parts.append(str(element))
    return ".".join(parts) if parts else "<suite>"


def _case_id_at(raw: object, loc: Sequence[object]) -> str | None:
    """Recover the id of the case a pydantic error points into, if it has one."""
    if not isinstance(raw, dict):
        return None
    cases = raw.get("cases")
    if len(loc) < 2 or loc[0] != "cases" or not isinstance(loc[1], int):  # noqa: PLR2004
        return None
    if not isinstance(cases, list) or not 0 <= loc[1] < len(cases):
        return None
    case = cases[loc[1]]
    if isinstance(case, dict):
        identifier = case.get("id")
        if isinstance(identifier, str):
            return identifier
    return None


def render_errors(file: str, errors: Sequence[FieldError]) -> str:
    """Build the multi-line message a :class:`BenchmarkValidationError` carries."""
    header = f"{file}: {len(errors)} validation error(s)"
    lines = [
        "  {location}{case}: {message} (input: {input_repr})".format(
            location=item.location,
            case="" if item.case_id is None else f" [case {item.case_id}]",
            message=item.message,
            input_repr=item.input_repr,
        )
        for item in errors
    ]
    return "\n".join([header, *lines])


def _file_level_error(
    file: str,
    message: str,
    *,
    location: str = "<file>",
    input_repr: str | None = None,
) -> BenchmarkValidationError:
    """Build a one-error :class:`BenchmarkValidationError` about the file itself."""
    errors = (
        FieldError(
            file=file,
            location=location,
            case_id=None,
            message=message,
            input_repr=input_repr,
        ),
    )
    return BenchmarkValidationError(render_errors(file, errors), errors=errors)


def _parse_document(path: Path, text: str) -> Any:
    """Parse `text` as safe YAML or JSON according to the file's extension.

    Raises:
        ValueError: for an unsupported extension.
        yaml.YAMLError | json.JSONDecodeError: for unparseable content.
    """
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        return json.loads(text)
    if suffix in _YAML_SUFFIXES:
        return yaml.safe_load(text)
    msg = f"unsupported suite file extension {suffix!r}; use .yaml, .yml or .json"
    raise ValueError(msg)


def read_suite_document(path: Path, *, max_bytes: int = DEFAULT_MAX_SUITE_BYTES) -> JSONValue:
    """Read and parse one suite file into plain JSON-compatible data.

    Raises:
        BenchmarkValidationError: when the file is missing, unreadable, not valid
            UTF-8, oversized, or not parseable as safe YAML or JSON. Every one of
            those is a problem with a FILE the user pointed at, so every one of
            them is a field error naming that file rather than a traceback.
    """
    file = str(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _file_level_error(file, f"cannot read benchmark file: {exc.strerror or exc}") from exc

    if size > max_bytes:
        raise _file_level_error(file, f"file is {size} bytes, over the {max_bytes} byte cap")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise _file_level_error(file, f"benchmark file is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise _file_level_error(file, f"cannot read benchmark file: {exc.strerror or exc}") from exc

    try:
        document = _parse_document(path, text)
    except ValueError as exc:
        # json.JSONDecodeError is a ValueError, as is the unsupported-extension
        # signal raised above; yaml raises its own base class.
        raise _file_level_error(file, str(exc)) from exc
    except yaml.YAMLError as exc:
        raise _file_level_error(file, f"could not be parsed as YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise _file_level_error(
            file,
            f"top level must be a mapping, got {type(document).__name__}",
            location="<suite>",
            input_repr=_truncate(document),
        )
    parsed: JSONValue = document
    return parsed


def load_suite(
    path: Path | str,
    *,
    root: Path | None = None,
    max_bytes: int = DEFAULT_MAX_SUITE_BYTES,
    validators: Sequence[SuiteValidator] = (),
) -> BenchmarkSuite:
    """Load, parse and fully validate one benchmark suite file.

    Validation is TWO STAGED, and the promise of "every error at once" holds
    within a stage rather than across both. Structural validation runs first;
    only if it passes do the injected `validators` run, because an
    evaluator-level check needs typed objects to inspect and cannot be given
    them until the structure is known to be sound.

    So a file with BOTH a duplicate case id and an unknown evaluator type
    reports the duplicate on the first attempt and the evaluator on the second.
    That is a real narrowing of the promise and it is stated here rather than
    discovered: the alternative is duplicating every evaluator check against raw
    dictionaries, which would give two implementations of one rule.

    Raises:
        BenchmarkValidationError: carrying one :class:`FieldError` per problem.
    """
    try:
        resolved = resolve_under_root(root, path)
    except PathEscapeError as exc:
        raise _file_level_error(str(path), str(exc)) from exc

    file = str(resolved)
    raw = read_suite_document(resolved, max_bytes=max_bytes)

    try:
        suite = BenchmarkSuite.model_validate(raw)
    except ValidationError as exc:
        errors = tuple(
            FieldError(
                file=file,
                location=format_location(detail["loc"]),
                case_id=_case_id_at(raw, detail["loc"]),
                message=detail["msg"],
                input_repr=_truncate(detail.get("input")),
            )
            for detail in exc.errors()
        )
        raise BenchmarkValidationError(render_errors(file, errors), errors=errors) from exc

    extra: list[FieldError] = []
    for validator in validators:
        extra.extend(validator(suite, file))
    if extra:
        found = tuple(extra)
        raise BenchmarkValidationError(render_errors(file, found), errors=found)

    return suite
