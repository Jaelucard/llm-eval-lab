"""Benchmark loading: every error at once, and no code execution from a file."""

from pathlib import Path

import pytest

from llm_eval_lab.datasets.loader import load_suite
from llm_eval_lab.evaluators import default_registry
from llm_eval_lab.models import BenchmarkValidationError

MIN_STRUCTURAL_ERRORS = 3
MIN_EVALUATOR_ERRORS = 3


def test_malformed_suite_reports_every_structural_error(fixtures_dir: Path) -> None:
    path = fixtures_dir / "suites" / "malformed_missing_id.yaml"
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(path)

    error = caught.value
    assert len(error.errors) >= MIN_STRUCTURAL_ERRORS
    locations = {item.location for item in error.errors}
    assert "cases[0].id" in locations
    assert any(location.startswith("cases[1]") for location in locations)
    assert any(location.startswith("cases[2]") for location in locations)

    message = str(error)
    assert str(path) in message
    assert "cases[0]" in message
    assert "has-neither" in message
    assert "has-unknown-field" in message


def test_malformed_suite_carries_the_failing_case_id(fixtures_dir: Path) -> None:
    path = fixtures_dir / "suites" / "malformed_missing_id.yaml"
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(path)

    by_case = {item.case_id for item in caught.value.errors}
    assert "has-neither" in by_case
    assert "has-unknown-field" in by_case


def test_evaluator_problems_are_load_time_errors(fixtures_dir: Path) -> None:
    path = fixtures_dir / "suites" / "malformed_bad_evaluator.yaml"
    registry = default_registry()
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(path, validators=(registry.validate_specs,))

    errors = caught.value.errors
    assert len(errors) >= MIN_EVALUATOR_ERRORS
    locations = {item.location for item in errors}
    assert "cases[0].evaluators[0].type" in locations
    assert "cases[1].evaluators[0].params.pattern" in locations
    assert "cases[2].evaluators[0].params.values" in locations

    messages = " ".join(item.message for item in errors)
    assert "unknown evaluator type" in messages
    assert "invalid regular expression" in messages


def test_python_object_yaml_tag_is_rejected_not_executed(tmp_path: Path) -> None:
    hostile = tmp_path / "hostile.yaml"
    marker = tmp_path / "executed.txt"
    hostile.write_text(
        "schema_version: 1\n"
        "name: hostile\n"
        "version: '1'\n"
        "cases: !!python/object/apply:os.system\n"
        f"  - touch {marker}\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(hostile)

    assert not marker.exists(), "safe_load must not execute a python object tag"
    assert "could not be parsed" in str(caught.value)


def test_a_valid_suite_loads(fixtures_dir: Path) -> None:
    suite = load_suite(fixtures_dir / "suites" / "minimal.yaml")
    assert suite.name == "minimal"
    assert len(suite.cases) == 1


def test_the_smoke_fixture_has_exactly_six_cases(fixtures_dir: Path) -> None:
    suite = load_suite(
        fixtures_dir / "suites" / "smoke.yaml",
        validators=(default_registry().validate_specs,),
    )
    assert len(suite.cases) == 6


def test_a_file_over_the_size_cap_is_refused(tmp_path: Path) -> None:
    big = tmp_path / "big.yaml"
    big.write_text("schema_version: 1\nname: big\ncases: []\n" + "#" * 4096, encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(big, max_bytes=64)
    assert "over the 64 byte cap" in str(caught.value)


def test_an_unsupported_extension_is_refused(tmp_path: Path) -> None:
    odd = tmp_path / "suite.txt"
    odd.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(odd)
    assert "unsupported suite file extension" in str(caught.value)


def test_a_path_escaping_the_configured_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "suites"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(outside, root=root)
    assert "outside the configured root" in str(caught.value)


def test_a_non_utf8_file_is_a_validation_error_not_a_traceback(tmp_path: Path) -> None:
    """A mis-encoded file is a problem with the file, so it must be a FieldError.

    `UnicodeDecodeError` is a `ValueError`, not an `LLMEvalError`, so before the
    read moved inside the guarded block it escaped the CLI's exception guard
    entirely and reached the user as a raw traceback with exit 1.
    """
    latin1 = tmp_path / "latin1.yaml"
    latin1.write_bytes(
        "schema_version: 1\nname: latin1\ndescription: caf\u00e9\n".encode("latin-1")
    )

    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(latin1)

    assert "not valid UTF-8" in str(caught.value)
    assert caught.value.errors[0].file == str(latin1)


def test_an_unreadable_file_is_a_validation_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(missing)
    assert "cannot read benchmark file" in str(caught.value)
