"""Content hashes for cases and suites.

A case hash exists so two runs of the same case can be paired for comparison,
and so a run stays comparable after its source file is re-tagged or
re-categorized. That is why :func:`compute_case_hash` covers ``id``,
``messages``, ``system``, ``expected``, ``params`` and ``evaluators`` and
deliberately excludes ``tags``, ``category``, ``metadata`` and ``weight``:
those describe how a case is filed, not what it asks or how it is judged.
Including them would break historical pairing every time someone tidied a
label.
"""

from llm_eval_lab.models import (
    ChatMessage,
    EvaluatorSpec,
    GenerationParams,
    JSONValue,
    ResolvedCase,
)
from llm_eval_lab.utils.hashing import HASH_PREFIX, canonical_digest
from llm_eval_lab.utils.json import canonical_json


def compute_case_hash(  # noqa: PLR0913 - the six inputs ARE the hashed surface; grouping
    # them into a struct would create a second place the exclusion rule is written.
    *,
    case_id: str,
    messages: tuple[ChatMessage, ...],
    system: str | None,
    expected: JSONValue,
    params: GenerationParams,
    evaluators: tuple[EvaluatorSpec, ...],
) -> str:
    """Return the content hash of one resolved case.

    Covers exactly what determines the case's behaviour. See the module
    docstring for why ``tags``, ``category``, ``metadata`` and ``weight`` are
    absent.
    """
    payload = {
        "id": case_id,
        "messages": [message.model_dump(mode="json") for message in messages],
        "system": system,
        "expected": expected,
        "params": params.model_dump(mode="json"),
        "evaluators": [spec.model_dump(mode="json") for spec in evaluators],
    }
    return canonical_digest(payload)


def compute_suite_hash(
    *,
    schema_version: int,
    name: str,
    version: str,
    cases: tuple[ResolvedCase, ...],
) -> str:
    """Return the content hash of a whole resolved suite.

    Case entries are sorted by id, so reordering cases in the source file does
    not change the digest. Only each case's id and hash contribute, so the
    suite hash inherits the case hash's exclusions for free.
    """
    payload = {
        "schema_version": schema_version,
        "name": name,
        "version": version,
        "cases": sorted(
            ({"id": case.id, "case_hash": case.case_hash} for case in cases),
            key=lambda entry: entry["id"],
        ),
    }
    return canonical_digest(payload)


def compute_content_hash(payload: object) -> str:
    """Return the content hash of any JSON-compatible payload.

    Used for the price table digest and the judge rubric digest, both of which
    need a stable identity for something that is not a case or a suite.
    """
    return canonical_digest(payload)


__all__ = [
    "HASH_PREFIX",
    "canonical_digest",
    "canonical_json",
    "compute_case_hash",
    "compute_content_hash",
    "compute_suite_hash",
]
