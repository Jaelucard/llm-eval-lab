"""The provider registry, credential resolution and the provider factory.

Providers are constructed through this module and nowhere else, which is what
gives credential handling a single implementation site. A
:class:`~llm_eval_lab.models.ProviderConfig` records the NAME of the
environment variable holding a credential, never the value; the resolver reads
the value on demand, wrapped in ``SecretStr``, and hands it straight to the
client. Nothing persisted, logged or returned over the API ever holds the
value itself.
"""

import importlib.util
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic import SecretStr

from llm_eval_lab.models import (
    MissingCredentialError,
    ProviderConfig,
    ProviderNotInstalledError,
)
from llm_eval_lab.providers.base import BaseProvider
from llm_eval_lab.providers.fake import FakeProvider
from llm_eval_lab.settings import read_credential


@dataclass(frozen=True)
class ProviderInfo:
    """The catalog entry for one registered provider.

    ``credential_env`` is the NAME of the environment variable a run defaults
    to when the caller passes no ``--api-key-env``. It is a name, never a
    value, which is what lets ``llm-eval models`` and ``llm-eval config show``
    tell an operator exactly what to set without ever reading it.

    ``models`` lists the model ids the provider is known to serve. It is a
    catalog aid rather than a whitelist: a provider is not required to refuse a
    model that is absent from it, because vendors add models faster than a
    shipped list can track them.

    ``import_name`` is the module whose presence decides whether the optional
    extra behind this provider is installed. It is a module path rather than the
    extra's name because the two differ: the ``google`` extra installs
    ``google-genai``, which imports as ``google.genai``. Recording it here is
    what lets :attr:`available` answer "can this build actually reach the
    vendor" without importing the SDK.
    """

    name: str
    requires_credential: bool
    extra: str | None
    summary: str
    models: tuple[str, ...] = ()
    credential_env: str | None = None
    import_name: str | None = None

    @property
    def available(self) -> bool:
        """Report whether this provider's backing SDK is importable in this build.

        Uses ``importlib.util.find_spec`` rather than an import, so asking the
        question costs no module execution and cannot fail a ``llm-eval
        providers`` listing because one vendor SDK happens to be broken. A
        provider with no ``import_name`` needs no extra and is always available.
        """
        if self.import_name is None:
            return True
        try:
            return importlib.util.find_spec(self.import_name) is not None
        except (ImportError, ValueError):
            # A missing parent package raises ModuleNotFoundError; a namespace
            # package with no spec raises ValueError. Both mean "not installed".
            return False


class MissingCredentialForVariableError(MissingCredentialError):
    """A named credential variable is unset, with the NAME as an attribute.

    The name has always been in the message. Carrying it as a field too is what
    will let the HTTP layer put it in a problem detail's ``env_var`` member
    without splitting the message on whitespace and hoping the wording never
    changes. That consumer does not exist yet: as of this phase the attribute is
    read only by tests, and wiring it into the API problem detail is a later
    phase's work. The alternative - adding it once the coupling had already been
    written as string-splitting - is how that coupling becomes permanent.

    A subclass rather than a new field on
    :class:`~llm_eval_lab.models.MissingCredentialError`, because the exception
    hierarchy is a frozen contract. Every existing ``except MissingCredentialError``
    still catches this.
    """

    env_var: str

    def __init__(self, env_var: str) -> None:
        """Name the variable that is unset. The VALUE is never touched."""
        self.env_var = env_var
        super().__init__(f"environment variable {env_var} is not set")


class ProviderExtraRequiredError(ProviderNotInstalledError):
    """A provider is registered but its optional install extra is absent.

    ``extra`` is the name a caller passes to ``pip install 'llm-eval-lab[...]'``.
    It is a package name, never a credential, and it is what will make a 424
    body actionable instead of merely negative. Like
    :attr:`MissingCredentialForVariableError.env_var`, that HTTP consumer is a
    later phase's work; here the attribute is read by tests and the name is also
    carried in the message.
    """

    extra: str

    def __init__(self, message: str, *, provider: str, extra: str) -> None:
        """Record the message plus the extra that would fix it."""
        self.extra = extra
        super().__init__(message, provider=provider)


class EnvCredentialResolver:
    """Resolves a credential environment-variable NAME to its value.

    The value never leaves a :class:`~pydantic.SecretStr`, and the error raised
    when a variable is unset names the variable and nothing else. That message
    reaches logs, CLI output and HTTP responses, so it must stay free of the
    value even in the failure path.
    """

    def resolve(self, env_var: str | None) -> SecretStr | None:
        """Return the credential named by `env_var`, or ``None`` when unnamed.

        Raises:
            MissingCredentialError: when a variable is named but unset or empty.
        """
        if env_var is None:
            return None
        value = read_credential(env_var)
        if value is None:
            raise MissingCredentialForVariableError(env_var)
        return value


type ProviderBuilder = Callable[[ProviderConfig, SecretStr | None], BaseProvider]
"""Constructs one provider from its configuration and its resolved credential."""


def build_fake_provider(config: ProviderConfig, credential: SecretStr | None) -> BaseProvider:
    """Build the fake provider. It needs no credential and ignores any it is given."""
    del credential
    return FakeProvider(config)


@dataclass(frozen=True)
class _Registration:
    """Internal record of one registered provider."""

    info: ProviderInfo
    builder: ProviderBuilder


class ProviderRegistry:
    """Maps a provider name to its builder, and resolves credentials for it."""

    def __init__(self, *, resolver: EnvCredentialResolver | None = None) -> None:
        """Create an empty registry with the given credential resolver."""
        self._registrations: dict[str, _Registration] = {}
        self._resolver = resolver or EnvCredentialResolver()

    def register(
        self,
        info: ProviderInfo,
        builder: ProviderBuilder,
    ) -> None:
        """Register one provider under `info.name`."""
        self._registrations[info.name] = _Registration(info=info, builder=builder)

    def names(self) -> tuple[str, ...]:
        """Return every registered provider name, sorted."""
        return tuple(sorted(self._registrations))

    def catalog(self) -> tuple[ProviderInfo, ...]:
        """Return the catalog entry for every registered provider."""
        return tuple(self._registrations[name].info for name in self.names())

    def info(self, name: str) -> ProviderInfo:
        """Return the catalog entry for one provider.

        Raises:
            ProviderNotInstalledError: when the name is not registered.
        """
        registration = self._registrations.get(name)
        if registration is None:
            msg = f"unknown provider {name!r}; registered providers are {', '.join(self.names())}"
            raise ProviderNotInstalledError(msg, provider=name)
        return registration.info

    def credential_status(self, name: str) -> bool | None:
        """Report whether a provider's default credential is present in the environment.

        Returns ``None`` for a provider that needs no credential, so "not
        required" is never confused with "required and missing". The value
        itself is neither returned nor logged; only whether it resolved.

        Raises:
            ProviderNotInstalledError: when the name is not registered.
        """
        info = self.info(name)
        if not info.requires_credential or info.credential_env is None:
            return None
        try:
            self._resolver.resolve(info.credential_env)
        except MissingCredentialError:
            return False
        return True

    def resolve_credential_env(self, config: ProviderConfig) -> str | None:
        """Return the environment variable NAME this configuration will read.

        A caller that never passed ``--api-key-env`` leaves
        :attr:`ProviderConfig.api_key_env` as ``None``, which is the ordinary
        ``llm-eval run --provider openai --model gpt-5`` path. Falling back to
        the registered default here is what stops that path from reaching the
        vendor SDK with ``api_key=None`` - because every SDK in this project
        then reads its own variable out of ``os.environ`` directly, which
        bypasses :func:`~llm_eval_lab.settings.read_credential`, bypasses the
        project's `.env` loading, makes
        :class:`MissingCredentialForVariableError` unreachable on the default
        path, and leaves the persisted run config claiming no variable was used
        when one was.

        A name, never a value. Returns ``None`` for a provider that needs no
        credential, so nothing is resolved for the fake, Ollama or
        OpenAI-compatible providers unless the caller named a variable itself.
        """
        if config.api_key_env is not None:
            return config.api_key_env
        info = self.info(config.provider)
        return info.credential_env if info.requires_credential else None

    def effective_config(self, config: ProviderConfig) -> ProviderConfig:
        """Return `config` with the credential variable NAME it actually uses recorded.

        The returned object is what gets persisted with the run, so a stored run
        says which variable paid for it rather than saying ``null`` and leaving
        the answer to whatever the environment happened to hold. Unchanged when
        the caller already named a variable, so the common path allocates
        nothing and the identity of the passed object is preserved.
        """
        resolved = self.resolve_credential_env(config)
        if resolved == config.api_key_env:
            return config
        return config.model_copy(update={"api_key_env": resolved})

    def ensure_credential(self, config: ProviderConfig) -> None:
        """Resolve a configuration's credential and discard it, or raise.

        The API launches a run as a background task, so anything that only fails
        once the provider is constructed fails AFTER the 202 has gone out: the
        client sees a run that quietly turns FAILED, and the name of the variable
        it needed to set reaches the server log and nowhere else. Calling this
        before the run row is written turns that into a 424 on the client's own
        request, which is what the API contract promises.

        The value is resolved and dropped. Nothing here returns, stores or logs
        it.

        Raises:
            ProviderNotInstalledError: when the provider name is not registered.
            MissingCredentialError: when a named credential variable is unset.
        """
        self.info(config.provider)
        self._resolver.resolve(self.resolve_credential_env(config))

    def create(self, config: ProviderConfig) -> BaseProvider:
        """Construct a provider, resolving its credential first.

        The credential is resolved HERE and handed to the builder explicitly, so
        no vendor SDK is ever given ``api_key=None`` and left to read the
        environment on its own. The provider is built from
        :meth:`effective_config`, so it also knows which variable it was given.

        The three checks run in order of how fundamental they are: the provider
        must exist, its SDK must be installed, and only then does a credential
        matter. Resolving the credential first would tell an operator with no
        vendor SDK installed to go and set an environment variable, which is not
        the thing standing between them and a working run.

        Raises:
            ProviderNotInstalledError: when the provider name is not registered,
                or when its optional extra is not installed.
            MissingCredentialError: when a named credential variable is unset,
                including the registered default a caller did not override.
        """
        registration = self._registrations.get(config.provider)
        if registration is None:
            msg = (
                f"unknown provider {config.provider!r}; registered providers are "
                f"{', '.join(self.names())}"
            )
            raise ProviderNotInstalledError(msg, provider=config.provider)
        require_extra(registration.info)
        effective = self.effective_config(config)
        credential = self._resolver.resolve(effective.api_key_env)
        return registration.builder(effective, credential)

    @asynccontextmanager
    async def open(self, config: ProviderConfig) -> AsyncIterator[BaseProvider]:
        """Yield a constructed provider and close it when the block exits."""
        provider = self.create(config)
        try:
            yield provider
        finally:
            await provider.aclose()


__all__ = [
    "ANTHROPIC_PROVIDER_INFO",
    "FAKE_MODELS",
    "FAKE_PROVIDER_INFO",
    "GOOGLE_PROVIDER_INFO",
    "OLLAMA_PROVIDER_INFO",
    "OPENAI_COMPATIBLE_PROVIDER_INFO",
    "OPENAI_PROVIDER_INFO",
    "REAL_PROVIDERS",
    "EnvCredentialResolver",
    "MissingCredentialForVariableError",
    "ProviderBuilder",
    "ProviderExtraRequiredError",
    "ProviderInfo",
    "ProviderRegistry",
    "build_anthropic_provider",
    "build_fake_provider",
    "build_google_provider",
    "build_ollama_provider",
    "build_openai_compatible_provider",
    "build_openai_provider",
    "require_extra",
]

FAKE_MODELS: tuple[str, ...] = ("fake-1",)
"""The fake provider's catalogued model.

It accepts any model id, because its output is derived from a hash of the
request rather than from a served model. One id is catalogued so that
``llm-eval models`` has something concrete to list, and it is the id the
example suite and the end-to-end test use.
"""

FAKE_PROVIDER_INFO = ProviderInfo(
    name=FakeProvider.name,
    requires_credential=False,
    extra=None,
    summary="Deterministic offline provider used by the test suite and by --provider fake.",
    models=FAKE_MODELS,
    credential_env=None,
)

# ---------------------------------------------------------------------------
# The real providers.
#
# Each entry pairs a catalog record with a builder that imports its adapter
# module only when a provider is actually constructed. Importing this registry
# therefore imports no vendor SDK, which is what keeps `llm-eval --help` cheap
# and what lets a build with no extras installed still LIST every provider and
# say honestly which of them it cannot reach.
#
# The model lists are transcribed from the vendors' current model and pricing
# pages on the date recorded in `docs/providers-sources.md`. They are a catalog
# aid, not a whitelist: any model id the vendor accepts works, listed or not.
# `ollama` and `openai_compatible` list none on purpose, because the models they
# serve are whatever the operator pulled or deployed, and a guess would be a
# fabrication rather than a convenience.
# ---------------------------------------------------------------------------


def require_extra(info: ProviderInfo) -> None:
    """Refuse to build a provider whose optional extra is not installed.

    Checked here, before the adapter module is imported, so the failure names
    the extra and the exact install command rather than surfacing as an
    ``ImportError`` from somewhere inside a vendor package.

    Raises:
        ProviderExtraRequiredError: when the backing SDK is not importable.
    """
    if info.available or info.extra is None:
        return
    msg = (
        f"the {info.name} provider needs the {info.extra!r} extra; install it with "
        f"pip install 'llm-eval-lab[{info.extra}]'"
    )
    raise ProviderExtraRequiredError(msg, provider=info.name, extra=info.extra)


OPENAI_PROVIDER_INFO = ProviderInfo(
    name="openai",
    requires_credential=True,
    extra="openai",
    summary="OpenAI models through the Responses API.",
    models=(
        "gpt-5.5",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ),
    credential_env="OPENAI_API_KEY",
    import_name="openai",
)

OPENAI_COMPATIBLE_PROVIDER_INFO = ProviderInfo(
    name="openai_compatible",
    requires_credential=False,
    extra="openai",
    summary=(
        "Any endpoint speaking the OpenAI Chat Completions format, reached through "
        "base_url. Requires base_url; a credential only if the endpoint checks one."
    ),
    models=(),
    credential_env=None,
    import_name="openai",
)

ANTHROPIC_PROVIDER_INFO = ProviderInfo(
    name="anthropic",
    requires_credential=True,
    extra="anthropic",
    summary="Anthropic models through the Messages API.",
    models=(
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
    ),
    credential_env="ANTHROPIC_API_KEY",
    import_name="anthropic",
)

GOOGLE_PROVIDER_INFO = ProviderInfo(
    name="google",
    requires_credential=True,
    extra="google",
    summary="Google Gemini models through the google-genai SDK.",
    models=(
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ),
    credential_env="GOOGLE_API_KEY",
    import_name="google.genai",
)

OLLAMA_PROVIDER_INFO = ProviderInfo(
    name="ollama",
    requires_credential=False,
    extra=None,
    summary=(
        "A local Ollama server through its native /api/chat endpoint. Needs no extra "
        "and no credential; defaults to http://localhost:11434."
    ),
    models=(),
    credential_env=None,
    import_name=None,
)


def build_openai_provider(config: ProviderConfig, credential: SecretStr | None) -> BaseProvider:
    """Build the OpenAI provider, importing its adapter only now.

    Raises:
        ProviderExtraRequiredError: when the ``openai`` extra is not installed.
    """
    require_extra(OPENAI_PROVIDER_INFO)
    from llm_eval_lab.providers.openai_provider import (  # noqa: PLC0415 - lazy by design
        OpenAIProvider,
    )

    return OpenAIProvider(config, credential)


def build_openai_compatible_provider(
    config: ProviderConfig, credential: SecretStr | None
) -> BaseProvider:
    """Build the OpenAI-compatible provider, importing its adapter only now.

    Raises:
        ProviderExtraRequiredError: when the ``openai`` extra is not installed.
        ProviderInvalidRequestError: when no ``base_url`` is configured.
    """
    require_extra(OPENAI_COMPATIBLE_PROVIDER_INFO)
    from llm_eval_lab.providers.openai_compatible import (  # noqa: PLC0415 - lazy by design
        OpenAICompatibleProvider,
    )

    return OpenAICompatibleProvider(config, credential)


def build_anthropic_provider(config: ProviderConfig, credential: SecretStr | None) -> BaseProvider:
    """Build the Anthropic provider, importing its adapter only now.

    Raises:
        ProviderExtraRequiredError: when the ``anthropic`` extra is not installed.
    """
    require_extra(ANTHROPIC_PROVIDER_INFO)
    from llm_eval_lab.providers.anthropic_provider import (  # noqa: PLC0415 - lazy by design
        AnthropicProvider,
    )

    return AnthropicProvider(config, credential)


def build_google_provider(config: ProviderConfig, credential: SecretStr | None) -> BaseProvider:
    """Build the Google provider, importing its adapter only now.

    Raises:
        ProviderExtraRequiredError: when the ``google`` extra is not installed.
    """
    require_extra(GOOGLE_PROVIDER_INFO)
    from llm_eval_lab.providers.google_provider import (  # noqa: PLC0415 - lazy by design
        GoogleProvider,
    )

    return GoogleProvider(config, credential)


def build_ollama_provider(config: ProviderConfig, credential: SecretStr | None) -> BaseProvider:
    """Build the Ollama provider. It needs no extra and usually no credential."""
    from llm_eval_lab.providers.ollama_provider import (  # noqa: PLC0415 - lazy by design
        OllamaProvider,
    )

    return OllamaProvider(config, credential)


REAL_PROVIDERS: tuple[tuple[ProviderInfo, ProviderBuilder], ...] = (
    (OPENAI_PROVIDER_INFO, build_openai_provider),
    (OPENAI_COMPATIBLE_PROVIDER_INFO, build_openai_compatible_provider),
    (ANTHROPIC_PROVIDER_INFO, build_anthropic_provider),
    (GOOGLE_PROVIDER_INFO, build_google_provider),
    (OLLAMA_PROVIDER_INFO, build_ollama_provider),
)
"""Every real provider, paired with its builder, in one list the registry walks."""
