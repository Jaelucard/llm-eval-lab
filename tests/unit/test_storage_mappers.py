"""`storage/mappers.py`'s scrub functions and the raw-payload size cap.

These three functions are the literal implementation of two named mitigations:
secret leakage into persisted rows (threat #2) and unbounded payload storage
(risk #11). Both are otherwise proven only end to end, through
`tests/integration/test_secret_leak.py`'s one canary scenario. This file pins
the boundary behaviour that scenario cannot reach on its own: absent or
malformed input shapes, the exact size-cap boundary, and idempotence.
"""

from typing import Any

from llm_eval_lab.redaction import REDACTED
from llm_eval_lab.storage.mappers import (
    RAW_TRUNCATED_KEY,
    cap_raw_payload,
    scrub_evaluation_payload,
    scrub_run_config,
)
from llm_eval_lab.utils.json import canonical_json_bytes

# ---------------------------------------------------------------------------
# scrub_run_config
# ---------------------------------------------------------------------------


def test_a_secret_shaped_key_in_params_extra_is_redacted() -> None:
    dump = {"params": {"temperature": 0.0, "extra": {"api_key": "sk-live-secret"}}}

    scrubbed = scrub_run_config(dump)

    assert scrubbed["params"]["extra"]["api_key"] == REDACTED
    assert scrubbed["params"]["temperature"] == 0.0, "a contract field beside it is untouched"


def test_a_secret_shaped_key_nested_inside_provider_options_is_redacted() -> None:
    """`options` nests arbitrarily deep (e.g. `headers.Authorization`), and the
    scrub must follow it there, not only at the top level of `options`."""
    dump = {
        "provider": {
            "provider": "openai_compatible",
            "options": {"headers": {"Authorization": "Bearer sk-live-secret"}, "mode": "chat"},
        }
    }

    scrubbed = scrub_run_config(dump)

    assert scrubbed["provider"]["options"]["headers"]["Authorization"] == REDACTED
    assert scrubbed["provider"]["options"]["mode"] == "chat"
    assert scrubbed["provider"]["provider"] == "openai_compatible", (
        "sibling contract fields on `provider` are untouched"
    )


def test_a_secret_shaped_key_in_an_evaluator_params_is_redacted() -> None:
    dump = {
        "evaluators": [
            {"type": "regex_match", "params": {"pattern": "^ok$"}},
            {"type": "llm_judge", "params": {"api_key": "sk-live-secret", "template_id": "t"}},
        ]
    }

    scrubbed = scrub_run_config(dump)

    assert scrubbed["evaluators"][0]["params"]["pattern"] == "^ok$"
    assert scrubbed["evaluators"][1]["params"]["api_key"] == REDACTED
    assert scrubbed["evaluators"][1]["params"]["template_id"] == "t"


def test_api_key_env_survives_because_it_names_a_variable_not_a_value() -> None:
    """`api_key_env` is the contract's legitimate way to record a credential's
    NAME. The `_env` suffix is what tells the scrub apart from `api_key`
    itself, and it must never be caught by the same substring match."""
    dump = {"provider": {"provider": "openai", "options": {"api_key_env": "OPENAI_API_KEY"}}}

    scrubbed = scrub_run_config(dump)

    assert scrubbed["provider"]["options"]["api_key_env"] == "OPENAI_API_KEY"


def test_scrub_run_config_leaves_contract_fields_outside_the_three_paths_alone() -> None:
    """Token counts and other contract-shaped integers must survive: scrubbing
    the whole config (rather than only `params.extra`, `provider.options` and
    `evaluators[i].params`) was tried and rejected, because it corrupts the
    very data it protects."""
    dump = {
        "suite_name": "smoke",
        "n_cases": 6,
        "params": {"max_output_tokens": 512},
        "provider": {"provider": "fake", "model": "fake-1"},
    }

    scrubbed = scrub_run_config(dump)

    assert scrubbed == dump


def test_scrub_run_config_tolerates_a_missing_params_extra() -> None:
    dump = {"params": {"temperature": 0.0}}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_params_extra_being_none() -> None:
    dump = {"params": {"temperature": 0.0, "extra": None}}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_a_missing_provider_options() -> None:
    dump = {"provider": {"provider": "fake", "model": "fake-1"}}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_provider_options_being_absent_entirely() -> None:
    dump: dict[str, Any] = {"suite_name": "smoke"}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_evaluators_being_absent() -> None:
    dump = {"suite_name": "smoke"}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_an_evaluator_spec_with_no_params() -> None:
    dump = {"evaluators": [{"type": "regex_match"}]}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_an_evaluator_spec_whose_params_is_not_a_dict() -> None:
    """A malformed spec (`params` not a mapping) is left alone rather than
    crashing the scrub - `ProviderConfig`/`EvaluatorSpec` validation is what
    rejects this shape, not the persistence-time scrub."""
    dump = {"evaluators": [{"type": "regex_match", "params": "not-a-dict"}]}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_tolerates_a_non_dict_entry_in_evaluators() -> None:
    dump = {"evaluators": ["not-a-dict-either"]}
    assert scrub_run_config(dump) == dump


def test_scrub_run_config_is_idempotent() -> None:
    dump = {
        "params": {"extra": {"api_key": "sk-live-secret"}},
        "provider": {"options": {"password": "hunter2"}},
        "evaluators": [{"type": "llm_judge", "params": {"secret": "s"}}],
    }

    once = scrub_run_config(dump)
    twice = scrub_run_config(once)

    assert twice == once


def test_scrub_run_config_does_not_mutate_its_input() -> None:
    dump = {"params": {"extra": {"api_key": "sk-live-secret"}}}
    original_extra = dump["params"]["extra"]

    scrub_run_config(dump)

    assert dump["params"]["extra"] is original_extra
    assert dump["params"]["extra"]["api_key"] == "sk-live-secret"


# ---------------------------------------------------------------------------
# scrub_evaluation_payload
# ---------------------------------------------------------------------------


def test_a_secret_shaped_key_in_metadata_is_redacted() -> None:
    dump = {
        "evaluator_id": "rubric-score-only",
        "metadata": {"template_id": "t", "judge_params": {"api_key": "sk-live-secret"}},
    }

    scrubbed = scrub_evaluation_payload(dump)

    assert scrubbed["metadata"]["judge_params"]["api_key"] == REDACTED
    assert scrubbed["metadata"]["template_id"] == "t"
    assert scrubbed["evaluator_id"] == "rubric-score-only"


def test_scrub_evaluation_payload_leaves_non_secret_metadata_untouched() -> None:
    dump = {"metadata": {"dispersion": 0.0, "n_judges": 2, "self_preference_risk": False}}
    assert scrub_evaluation_payload(dump) == dump


def test_scrub_evaluation_payload_tolerates_a_missing_metadata() -> None:
    dump = {"evaluator_id": "regex-match"}
    assert scrub_evaluation_payload(dump) == dump


def test_scrub_evaluation_payload_tolerates_metadata_being_none() -> None:
    dump = {"evaluator_id": "regex-match", "metadata": None}
    assert scrub_evaluation_payload(dump) == dump


def test_scrub_evaluation_payload_tolerates_metadata_not_being_a_dict() -> None:
    dump = {"evaluator_id": "regex-match", "metadata": "not-a-dict"}
    assert scrub_evaluation_payload(dump) == dump


def test_scrub_evaluation_payload_is_idempotent() -> None:
    dump = {"metadata": {"token": "s", "nested": {"password": "p"}}}

    once = scrub_evaluation_payload(dump)
    twice = scrub_evaluation_payload(once)

    assert twice == once


# ---------------------------------------------------------------------------
# cap_raw_payload
# ---------------------------------------------------------------------------


def test_a_payload_under_the_cap_is_returned_unchanged() -> None:
    raw = {"output": "hello world"}
    assert cap_raw_payload(raw, max_bytes=len(canonical_json_bytes(raw)) + 1) == raw


def test_a_payload_exactly_at_the_cap_is_returned_unchanged() -> None:
    """The boundary is inclusive: `size <= max_bytes` keeps a payload whose
    canonical encoding lands EXACTLY on the configured cap."""
    raw = {"output": "exactly this many bytes"}
    exact_size = len(canonical_json_bytes(raw))

    assert cap_raw_payload(raw, max_bytes=exact_size) == raw


def test_a_payload_one_byte_over_the_cap_is_truncated() -> None:
    raw = {"output": "exactly this many bytes plus one"}
    exact_size = len(canonical_json_bytes(raw))

    capped = cap_raw_payload(raw, max_bytes=exact_size - 1)

    assert capped[RAW_TRUNCATED_KEY] is True
    assert capped["reason"] == "payload exceeded max_raw_bytes"
    assert capped["bytes"] == exact_size


def test_a_grossly_oversized_payload_is_truncated_with_its_real_size() -> None:
    raw = {"output": "x" * 10_000}
    capped = cap_raw_payload(raw, max_bytes=100)

    assert capped[RAW_TRUNCATED_KEY] is True
    assert capped["reason"] == "payload exceeded max_raw_bytes"
    assert capped["bytes"] > 100


def test_a_non_json_serializable_payload_is_truncated_for_that_reason() -> None:
    """`{1, 2, 3}` is a `set`; `json.dumps` has no representation for one, and
    that `TypeError` must become a marker rather than propagate and abort the
    write of an otherwise-good case result."""
    raw = {"bad": {1, 2, 3}}

    capped = cap_raw_payload(raw, max_bytes=1_000_000)

    assert capped[RAW_TRUNCATED_KEY] is True
    assert capped["reason"] == "payload is not JSON-serializable"
    assert "bytes" not in capped, "the size could not be computed, so none is claimed"


def test_a_nan_valued_payload_is_truncated_for_the_same_reason() -> None:
    """`canonical_json` calls `json.dumps(..., allow_nan=False)`, which raises
    `ValueError` (not `TypeError`) for a NaN - both must be caught."""
    raw = {"bad": float("nan")}

    capped = cap_raw_payload(raw, max_bytes=1_000_000)

    assert capped[RAW_TRUNCATED_KEY] is True
    assert capped["reason"] == "payload is not JSON-serializable"


def test_cap_raw_payload_is_idempotent_once_truncated() -> None:
    """A truncation marker is itself a small, JSON-serializable dict, so capping
    an already-capped payload again must not error and must produce the same
    marker shape (re-truncated for its own, now-different reason and size)."""
    raw = {"output": "x" * 10_000}
    once = cap_raw_payload(raw, max_bytes=100)
    twice = cap_raw_payload(once, max_bytes=100)

    assert twice[RAW_TRUNCATED_KEY] is True
    assert twice["reason"] == "payload exceeded max_raw_bytes"


def test_cap_raw_payload_does_not_mutate_its_input() -> None:
    raw = {"output": "hello"}
    cap_raw_payload(raw, max_bytes=1)
    assert raw == {"output": "hello"}
