"""Case and suite hashing: what they cover, and what they deliberately do not."""

from typing import Any

from llm_eval_lab.datasets.hashing import compute_case_hash, compute_suite_hash
from llm_eval_lab.datasets.resolve import resolve_suite
from llm_eval_lab.models import (
    BenchmarkSuite,
    ChatMessage,
    EvaluatorSpec,
    GenerationParams,
)
from llm_eval_lab.utils.json import canonical_json


def _case(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "c1",
        "input": "What is the capital of France?",
        "expected": "Paris",
        "evaluators": [{"type": "exact_match"}],
    }
    return {**base, **overrides}


def _suite(*cases: dict[str, Any], **overrides: Any) -> BenchmarkSuite:
    document: dict[str, Any] = {
        "schema_version": 1,
        "name": "hashing",
        "version": "1",
        "cases": list(cases),
        **overrides,
    }
    return BenchmarkSuite.model_validate(document)


def _hash_of(**overrides: Any) -> str:
    resolved = resolve_suite(_suite(_case(**overrides)))
    return resolved.cases[0].case_hash


def test_case_hash_ignores_tags_category_metadata_and_weight() -> None:
    baseline = _hash_of()
    assert _hash_of(tags=["regression", "smoke"]) == baseline
    assert _hash_of(category="geography") == baseline
    assert _hash_of(metadata={"owner": "someone"}) == baseline
    assert _hash_of(weight=3.5) == baseline


def test_case_hash_changes_when_expected_changes() -> None:
    assert _hash_of(expected="Lyon") != _hash_of()


def test_case_hash_changes_when_params_change() -> None:
    assert _hash_of(params={"temperature": 0.7}) != _hash_of()


def test_case_hash_changes_when_evaluators_change() -> None:
    assert _hash_of(evaluators=[{"type": "exact_match_ci"}]) != _hash_of()


def test_case_hash_is_prefixed() -> None:
    assert _hash_of().startswith("sha256:")


def test_suite_hash_is_invariant_to_case_ordering() -> None:
    forwards = resolve_suite(_suite(_case(id="a"), _case(id="b"), _case(id="c")))
    backwards = resolve_suite(_suite(_case(id="c"), _case(id="b"), _case(id="a")))
    assert forwards.suite_hash == backwards.suite_hash


def test_suite_hash_is_invariant_to_key_ordering_in_the_source() -> None:
    ordered = _suite({"id": "a", "input": "hello", "expected": "hi"})
    shuffled = _suite({"expected": "hi", "input": "hello", "id": "a"})
    assert resolve_suite(ordered).suite_hash == resolve_suite(shuffled).suite_hash


def test_suite_hash_changes_when_a_case_changes() -> None:
    original = resolve_suite(_suite(_case(id="a"), _case(id="b")))
    edited = resolve_suite(_suite(_case(id="a"), _case(id="b", expected="different")))
    assert original.suite_hash != edited.suite_hash


def test_canonical_json_sorts_keys_and_is_compact() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_compute_case_hash_is_stable_across_equal_inputs() -> None:
    kwargs: dict[str, Any] = {
        "case_id": "c1",
        "messages": (ChatMessage(role="user", content="hello"),),
        "system": None,
        "expected": "hi",
        "params": GenerationParams(),
        "evaluators": (EvaluatorSpec(type="exact_match"),),
    }
    assert compute_case_hash(**kwargs) == compute_case_hash(**kwargs)


def test_compute_suite_hash_covers_the_declared_identity_fields() -> None:
    resolved = resolve_suite(_suite(_case()))
    same = compute_suite_hash(schema_version=1, name="hashing", version="1", cases=resolved.cases)
    renamed = compute_suite_hash(schema_version=1, name="other", version="1", cases=resolved.cases)
    assert same == resolved.suite_hash
    assert renamed != resolved.suite_hash
