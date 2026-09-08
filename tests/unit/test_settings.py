"""Tests for the one module allowed to read the environment.

`settings.py` is frozen at the contract gate, so its defaults become expensive
to correct the moment the gate closes. The security-relevant ones are pinned
here: plugins off, loopback bind, no wildcard CORS origin, prompts unlogged, and
no vendor credential anywhere in the model. The last of those is asserted
mechanically, over the field annotations, rather than by reading the class.

No test in this file touches the repository's own `.env`. Environment values
come from `monkeypatch.setenv`, and the file-loading path is exercised through
an explicit `_env_file` override pointing inside `tmp_path`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr, ValidationError

from llm_eval_lab.models import EvaluatorSettings
from llm_eval_lab.settings import (
    Settings,
    _dotenv_path,
    _load_dotenv_once,
    get_settings,
    read_credential,
)

if TYPE_CHECKING:
    from pathlib import Path

# A credential-shaped value that is not a credential. It exists to be asserted
# absent from reprs, never to authenticate anything.
_FAKE_SECRET = "sk-not-a-real-key-0123456789"  # noqa: S105 - a decoy asserted absent from reprs, never a credential


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited LLM_EVAL_* variables so defaults are actually the defaults."""
    for name in list(os.environ):
        if name.startswith("LLM_EVAL_"):
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    _load_dotenv_once.cache_clear()
    _dotenv_path.cache_clear()


def _settings() -> Settings:
    """Construct Settings without reading any dotenv file."""
    return Settings(_env_file=None)


# --- security-relevant defaults ------------------------------------------


def test_plugins_are_disabled_by_default() -> None:
    """Addendum 3: loading a third-party entry point executes its code."""
    assert _settings().plugins_enabled is False


def test_no_plugin_is_allowed_by_default() -> None:
    assert _settings().plugins_allowed == frozenset()


def test_api_binds_to_loopback_by_default() -> None:
    settings = _settings()
    assert settings.api_host == "127.0.0.1"
    assert settings.api_bind_is_loopback is True


def test_a_non_loopback_bind_is_visible_to_the_startup_check() -> None:
    assert Settings(_env_file=None, api_host="0.0.0.0").api_bind_is_loopback is False  # noqa: S104 - asserting the check fires, not binding


@pytest.mark.parametrize("value", ["", " ", "\t", "\n  \t"])
def test_a_set_but_empty_api_token_is_normalised_to_none(value: str) -> None:
    """S-1: an empty token used to disable BOTH controls it configures.

    `export LLM_EVAL_API_TOKEN=$UNSET_VAR`, a `.env` line reading
    `LLM_EVAL_API_TOKEN=`, or a secrets mount that produced an empty file all
    yield `SecretStr('')`, which is not `None`. The bind check then accepted a
    non-loopback address because a token "was set", and the bearer gate compared
    the empty presented value against the empty configured one and matched - so a
    request with no `Authorization` header at all was authorised. Collapsing the
    two spellings into one state is what stops that.
    """
    assert Settings(_env_file=None, api_token=value).api_token is None


def test_a_real_api_token_survives_normalisation() -> None:
    settings = Settings(_env_file=None, api_token="  a-real-token  ")  # noqa: S106 - a test token
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == "  a-real-token  ", (
        "the value is normalised for emptiness only; it is never trimmed, "
        "because whitespace inside a token an operator configured is part of it"
    )


def test_an_empty_token_from_the_environment_is_normalised_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path that actually produces this in the wild is the environment."""
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", "")
    assert Settings(_env_file=None).api_token is None


def test_cors_origins_never_defaults_to_a_wildcard() -> None:
    assert "*" not in _settings().cors_origins


def test_cors_origins_rejects_a_wildcard_outright() -> None:
    """`POST /api/runs` spends money, so no configuration may open it to any origin."""
    with pytest.raises(ValidationError, match=r"must not contain"):
        Settings(_env_file=None, cors_origins=("*",))


def test_prompts_are_not_logged_by_default() -> None:
    """Prompt and output text are user data."""
    assert _settings().log_prompts is False


def test_env_prefix_is_the_documented_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "17")
    assert _settings().concurrency == 17


# --- credentials are not settings fields ----------------------------------


def test_api_token_is_the_only_secret_field() -> None:
    """The mechanical form of "credentials are not configuration".

    Any future field annotated `SecretStr` is a credential that would then be
    persisted with the settings object, so this test is what stops one appearing.
    """
    secret_fields = sorted(
        name
        for name, field in Settings.model_fields.items()
        if field.annotation in (SecretStr, SecretStr | None)
    )
    assert secret_fields == ["api_token"]


def test_no_vendor_credential_field_exists() -> None:
    forbidden = ("api_key", "apikey", "token", "secret", "password", "credential")
    offenders = sorted(
        name
        for name in Settings.model_fields
        if any(t in name.lower() for t in forbidden) and name != "api_token"
    )
    assert offenders == []


def test_api_token_value_does_not_leak_through_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", _FAKE_SECRET)
    settings = _settings()
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == _FAKE_SECRET
    assert _FAKE_SECRET not in repr(settings)
    assert _FAKE_SECRET not in settings.model_dump_json()


# --- read_credential -------------------------------------------------------


def test_read_credential_returns_the_value_as_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VENDOR_KEY_FOR_TEST", _FAKE_SECRET)
    credential = read_credential("VENDOR_KEY_FOR_TEST")
    assert credential is not None
    assert credential.get_secret_value() == _FAKE_SECRET


def test_read_credential_does_not_expose_the_value_in_its_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VENDOR_KEY_FOR_TEST", _FAKE_SECRET)
    assert _FAKE_SECRET not in repr(read_credential("VENDOR_KEY_FOR_TEST"))


def test_read_credential_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VENDOR_KEY_FOR_TEST", raising=False)
    assert read_credential("VENDOR_KEY_FOR_TEST") is None


def test_read_credential_treats_an_empty_value_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported-but-empty variable is a missing credential, not an empty one."""
    monkeypatch.setenv("VENDOR_KEY_FOR_TEST", "")
    assert read_credential("VENDOR_KEY_FOR_TEST") is None


def test_read_credential_never_stores_the_value_on_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VENDOR_KEY_FOR_TEST", _FAKE_SECRET)
    assert _FAKE_SECRET not in _settings().model_dump_json()


# --- dotenv support --------------------------------------------------------


def test_settings_declares_a_dotenv_file() -> None:
    """The LLM_EVAL_* half of the .env story.

    `endswith` rather than equality: the path is resolved with
    `find_dotenv(usecwd=True)`, so it is absolute wherever a file was found and
    the bare relative fallback only where none was.
    """
    env_file = Settings.model_config.get("env_file")
    assert env_file is not None
    assert str(env_file).endswith(".env")


def test_settings_reads_llm_eval_values_from_an_env_file(tmp_path: Path) -> None:
    """Exercised through an explicit override, never the repository's own file."""
    env_file = tmp_path / "settings.env"
    env_file.write_text("LLM_EVAL_CONCURRENCY=23\nLLM_EVAL_LOG_LEVEL=DEBUG\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)
    assert settings.concurrency == 23
    assert settings.log_level == "DEBUG"


def test_a_real_environment_variable_beats_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "settings.env"
    env_file.write_text("LLM_EVAL_CONCURRENCY=23\n", encoding="utf-8")
    monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "31")

    assert Settings(_env_file=env_file).concurrency == 31


# --- projections and derived values ---------------------------------------


def test_evaluator_settings_projects_the_four_fields_it_claims_to() -> None:
    settings = Settings(
        _env_file=None,
        judge_concurrency=9,
        max_output_chars=1234,
        max_regex_match_s=3.5,
        max_raw_bytes=4096,
    )
    projected = settings.evaluator_settings()

    assert projected == EvaluatorSettings(
        judge_concurrency=9,
        max_output_chars=1234,
        max_regex_match_s=3.5,
        max_raw_bytes=4096,
    )


def test_database_url_is_used_verbatim_when_given() -> None:
    url = "postgresql+asyncpg://localhost/evals"
    assert Settings(_env_file=None, database_url=url).resolved_database_url() == url


def test_database_url_falls_back_to_a_sqlite_file_under_the_data_dir(tmp_path: Path) -> None:
    resolved = Settings(_env_file=None, data_dir=tmp_path).resolved_database_url()
    assert resolved.startswith("sqlite+aiosqlite:///")
    assert str(tmp_path) in resolved


def test_env_file_default_stays_a_relative_literal_not_an_import_time_resolution() -> None:
    """`model_config["env_file"]` must never be replaced by an eagerly-resolved path.

    That was the earlier shape: a module-level constant calling
    `find_dotenv(usecwd=True)` at IMPORT time, baked straight into
    `SettingsConfigDict`. It could never be corrected afterward - there was no
    cache to clear - and it made importing this module walk the filesystem
    for a value most processes never need. `_dotenv_path()` now does that
    resolution instead, lazily and cached, and `get_settings()` passes it
    explicitly as `_env_file`; the class's own default has nothing to do
    except stay a plain, unresolved relative path.
    """
    assert Settings.model_config.get("env_file") == ".env"


def test_dotenv_path_is_cached_and_resolves_lazily() -> None:
    """Resolved on first call, not at import time, and stable until cleared."""
    first = _dotenv_path()
    second = _dotenv_path()
    assert first == second, "cached: two calls in the same process agree"
    assert isinstance(first, str)


def test_settings_are_frozen() -> None:
    """A component handed a Settings cannot reconfigure the process behind its caller."""
    settings = _settings()
    with pytest.raises(ValidationError):
        # setattr, not direct assignment: mypy knows the model is frozen, and
        # rejecting the assignment statically would hide the runtime assertion.
        setattr(settings, "concurrency", 99)  # noqa: B010 - see above


def test_one_dotenv_file_feeds_both_credentials_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two halves of the .env story, joined.

    `read_credential` reads `os.environ` and `Settings` reads its own fields, so
    a single file has to reach both. It also has to work for a resolver built
    before anything touches `get_settings`, which is why `read_credential` is
    called first here.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_EVAL_LOG_LEVEL", raising=False)
    assert read_credential("OPENAI_API_KEY") is None, "precondition: the key must start unset"

    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-value-not-real\nLLM_EVAL_LOG_LEVEL=DEBUG\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _dotenv_path.cache_clear()
    _load_dotenv_once.cache_clear()
    get_settings.cache_clear()

    credential = read_credential("OPENAI_API_KEY")
    assert credential is not None
    assert credential.get_secret_value() == "test-value-not-real"

    assert get_settings().log_level == "DEBUG"


def test_the_dotenv_load_never_overrides_an_existing_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`override=False`: a real environment variable always beats the file."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    monkeypatch.chdir(tmp_path)
    _dotenv_path.cache_clear()
    _load_dotenv_once.cache_clear()

    credential = read_credential("OPENAI_API_KEY")
    assert credential is not None
    assert credential.get_secret_value() == "from-the-environment"
