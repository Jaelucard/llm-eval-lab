"""The provider catalog, credential resolution and the missing-extra failure path."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from llm_eval_lab.models import (
    MissingCredentialError,
    ProviderConfig,
    ProviderNotInstalledError,
)
from llm_eval_lab.providers import default_registry
from llm_eval_lab.providers.registry import (
    MissingCredentialForVariableError,
    ProviderExtraRequiredError,
    ProviderRegistry,
)
from llm_eval_lab.providers.remote import RemoteProvider

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec

EXPECTED_PROVIDERS = (
    "anthropic",
    "fake",
    "google",
    "ollama",
    "openai",
    "openai_compatible",
)

CREDENTIAL_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "fake": None,
    "ollama": None,
    "openai_compatible": None,
}

REQUIRED_EXTRAS = {
    "openai": "openai",
    "openai_compatible": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "fake": None,
    "ollama": None,
}

IMPORT_NAMES = {
    "openai": "openai",
    "openai_compatible": "openai",
    "anthropic": "anthropic",
    "google": "google.genai",
}


# --- catalog ---------------------------------------------------------------


def test_every_provider_is_registered() -> None:
    assert default_registry().names() == EXPECTED_PROVIDERS


@pytest.mark.parametrize("name", EXPECTED_PROVIDERS)
def test_each_catalog_entry_names_its_extra_and_its_credential_variable(name: str) -> None:
    info = default_registry().info(name)

    assert info.extra == REQUIRED_EXTRAS[name]
    assert info.credential_env == CREDENTIAL_ENVS[name]
    assert info.requires_credential is (CREDENTIAL_ENVS[name] is not None)
    assert info.summary


@pytest.mark.parametrize("name", EXPECTED_PROVIDERS)
def test_availability_is_reported_for_every_provider(name: str) -> None:
    """All extras are installed in the development environment, so all are available."""
    assert default_registry().info(name).available is True


def test_a_provider_with_no_extra_is_always_available() -> None:
    for name in ("fake", "ollama"):
        info = default_registry().info(name)
        assert info.import_name is None
        assert info.available is True


def test_availability_is_false_when_the_backing_module_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str, package: str | None = None) -> ModuleSpec | None:
        del package
        return None if name == "openai" else object()  # type: ignore[return-value]

    monkeypatch.setattr("importlib.util.find_spec", missing)

    assert default_registry().info("openai").available is False
    assert default_registry().info("anthropic").available is True


def test_availability_survives_a_missing_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`find_spec("google.genai")` raises when `google` itself is absent."""

    def exploding(name: str, package: str | None = None) -> ModuleSpec | None:
        del package
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.util.find_spec", exploding)

    assert default_registry().info("google").available is False


def test_the_catalog_lists_models_where_a_shipped_list_is_meaningful() -> None:
    catalog = {info.name: info for info in default_registry().catalog()}

    assert "claude-sonnet-5" in catalog["anthropic"].models
    assert "gpt-5.1" in catalog["openai"].models
    assert "gemini-2.5-flash" in catalog["google"].models
    assert catalog["ollama"].models == (), "the served models are whatever the operator pulled"
    assert catalog["openai_compatible"].models == (), "the served models depend on the endpoint"


# --- missing extras --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("openai", "openai"),
        ("openai_compatible", "openai"),
        ("anthropic", "anthropic"),
        ("google", "google"),
    ],
)
def test_creating_a_provider_whose_extra_is_absent_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, name: str, extra: str
) -> None:
    def missing(module: str, package: str | None = None) -> ModuleSpec | None:
        del package
        return None if module == IMPORT_NAMES[name] else object()  # type: ignore[return-value]

    monkeypatch.setattr("importlib.util.find_spec", missing)

    with pytest.raises(ProviderNotInstalledError) as caught:
        default_registry().create(
            ProviderConfig(provider=name, model="m", base_url="http://localhost:8000/v1")
        )

    assert isinstance(caught.value, ProviderExtraRequiredError)
    assert caught.value.extra == extra
    assert extra in caught.value.message
    assert f"llm-eval-lab[{extra}]" in caught.value.message


def test_an_sdk_that_fails_to_import_is_still_reported_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second net: a spec that exists but an import that fails."""
    from llm_eval_lab.providers.openai_provider import import_openai  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ProviderExtraRequiredError) as caught:
        import_openai("openai")

    assert caught.value.extra == "openai"


def test_an_unknown_provider_name_lists_the_ones_that_exist() -> None:
    with pytest.raises(ProviderNotInstalledError) as caught:
        default_registry().create(ProviderConfig(provider="nope", model="m"))

    assert "nope" in str(caught.value)
    assert "openai" in str(caught.value)


# --- credentials -----------------------------------------------------------


def test_a_missing_credential_names_the_variable_and_never_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ProviderConfig(provider="openai", model="gpt-5.1", api_key_env="OPENAI_API_KEY")

    with pytest.raises(MissingCredentialError) as caught:
        default_registry().create(config)

    assert "OPENAI_API_KEY" in str(caught.value)


def test_a_credential_value_never_appears_in_the_failure_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variable is SET, then a different one is requested. Neither value leaks."""
    planted = "test-value-that-must-not-be-echoed"
    monkeypatch.setenv("OPENAI_API_KEY", planted)
    monkeypatch.delenv("SOME_OTHER_KEY_ENV", raising=False)
    config = ProviderConfig(provider="openai", model="gpt-5.1", api_key_env="SOME_OTHER_KEY_ENV")

    with pytest.raises(MissingCredentialError) as caught:
        default_registry().create(config)

    assert planted not in str(caught.value)
    assert "SOME_OTHER_KEY_ENV" in str(caught.value)


# --- the default credential variable, when the caller named none ------------


@pytest.mark.parametrize(
    ("name", "env_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
    ],
)
def test_an_unset_default_variable_is_reported_by_name_when_api_key_env_is_none(
    monkeypatch: pytest.MonkeyPatch, name: str, env_var: str
) -> None:
    """`llm-eval run --provider openai` names no variable; the default must still resolve.

    Without the fallback this reaches the vendor SDK with `api_key=None` and the
    SDK reads its own variable from the environment, which bypasses
    `settings.read_credential` and makes this error unreachable.
    """
    monkeypatch.delenv(env_var, raising=False)
    config = ProviderConfig(provider=name, model="m")

    with pytest.raises(MissingCredentialForVariableError) as caught:
        default_registry().create(config)

    assert caught.value.env_var == env_var
    assert env_var in str(caught.value)


@pytest.mark.parametrize(
    ("name", "env_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
    ],
)
def test_a_set_default_variable_is_resolved_and_passed_explicitly(
    monkeypatch: pytest.MonkeyPatch, name: str, env_var: str
) -> None:
    planted = "test-value-not-a-real-credential"
    monkeypatch.setenv(env_var, planted)

    provider = default_registry().create(ProviderConfig(provider=name, model="m"))

    assert isinstance(provider, RemoteProvider)
    assert provider._api_key() == planted, "the key must be handed to the SDK, not left to it"


@pytest.mark.parametrize(
    ("name", "env_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
    ],
)
def test_the_variable_that_paid_for_the_run_is_recorded_in_the_effective_config(
    monkeypatch: pytest.MonkeyPatch, name: str, env_var: str
) -> None:
    """A persisted run that says api_key_env=None cannot be reproduced."""
    monkeypatch.setenv(env_var, "test-value-not-a-real-credential")

    provider = default_registry().create(ProviderConfig(provider=name, model="m"))

    assert provider.config.api_key_env == env_var


def test_an_explicitly_named_variable_is_never_replaced_by_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_OWN_KEY_ENV", "test-value-not-a-real-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "a-different-value-that-must-not-win")
    config = ProviderConfig(provider="openai", model="m", api_key_env="MY_OWN_KEY_ENV")

    provider = default_registry().create(config)

    assert provider.config is config, "the common path allocates no copy"
    assert isinstance(provider, RemoteProvider)
    assert provider._api_key() == "test-value-not-a-real-credential"


@pytest.mark.parametrize("name", ["fake", "ollama", "openai_compatible"])
def test_a_provider_needing_no_credential_gets_no_default_variable(name: str) -> None:
    config = ProviderConfig(
        provider=name,
        model="m",
        base_url="http://localhost:8000/v1" if name == "openai_compatible" else None,
    )
    registry = default_registry()

    assert registry.resolve_credential_env(config) is None
    assert registry.effective_config(config) is config


def test_ensure_credential_checks_the_default_variable_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API calls this before writing the run row, so a 424 beats a silent FAILED."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingCredentialForVariableError) as caught:
        default_registry().ensure_credential(ProviderConfig(provider="openai", model="m"))

    assert caught.value.env_var == "OPENAI_API_KEY"


def test_a_missing_extra_is_reported_before_a_missing_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telling someone to set a variable when the SDK is absent is the wrong advice."""

    def missing(module: str, package: str | None = None) -> ModuleSpec | None:
        del package
        return None if module == "openai" else object()  # type: ignore[return-value]

    monkeypatch.setattr("importlib.util.find_spec", missing)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderExtraRequiredError):
        default_registry().create(ProviderConfig(provider="openai", model="m"))


def test_credential_status_distinguishes_not_required_from_required_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-value-not-a-real-credential")
    registry = default_registry()

    assert registry.credential_status("ollama") is None
    assert registry.credential_status("anthropic") is False
    assert registry.credential_status("openai") is True


def test_a_resolved_credential_reaches_the_constructed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-value-not-a-real-credential")
    config = ProviderConfig(provider="openai", model="gpt-5.1", api_key_env="OPENAI_API_KEY")

    provider = default_registry().create(config)

    assert provider.name == "openai"
    assert provider.config is config


# --- construction ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("fake", {}),
        ("ollama", {}),
        ("openai_compatible", {"base_url": "http://localhost:8000/v1"}),
    ],
)
def test_credential_free_providers_construct_without_any_environment(
    name: str, overrides: dict[str, Any]
) -> None:
    provider = default_registry().create(ProviderConfig(provider=name, model="m", **overrides))

    assert provider.name == name


async def test_open_closes_the_provider_when_the_block_exits() -> None:
    config = ProviderConfig(provider="ollama", model="llama3.2")

    async with default_registry().open(config) as provider:
        opened = provider.client()  # type: ignore[attr-defined]
        assert opened.is_closed is False

    assert opened.is_closed is True


def test_the_registry_is_a_process_wide_singleton() -> None:
    assert default_registry() is default_registry()


def test_an_empty_registry_registers_nothing_by_accident() -> None:
    assert ProviderRegistry().names() == ()


# --- lazy imports ----------------------------------------------------------

PROBE = """
import sys

from llm_eval_lab.providers import default_registry

registry = default_registry()
assert len(registry.names()) == 6, registry.names()
watched = ("openai", "anthropic", "google.genai")
print(",".join(sorted(name for name in watched if name in sys.modules)))
"""


@pytest.mark.integration
def test_building_the_registry_imports_no_vendor_sdk() -> None:
    """Listing providers must not cost the import of three vendor packages."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-controlled input
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"registry import loaded {completed.stdout.strip()}"
