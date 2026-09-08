"""Provider integrations and the registry that constructs them.

:func:`default_registry` is the process-wide registry. It is built lazily and
cached, so importing this package never imports a vendor SDK: a provider that
needs an optional extra registers a builder that imports it on first
construction, not at module import time.
"""

from functools import lru_cache

from llm_eval_lab.providers.base import (
    ATTEMPT_METADATA_KEY,
    CASE_ID_METADATA_KEY,
    EXPECTED_METADATA_KEY,
    RUN_ID_METADATA_KEY,
    BaseProvider,
)
from llm_eval_lab.providers.fake import FakeProvider, FakeProviderOptions
from llm_eval_lab.providers.registry import (
    FAKE_PROVIDER_INFO,
    REAL_PROVIDERS,
    EnvCredentialResolver,
    ProviderInfo,
    ProviderRegistry,
    build_fake_provider,
)


@lru_cache(maxsize=1)
def default_registry() -> ProviderRegistry:
    """Return the process-wide registry of available providers.

    Every provider is registered, including ones whose optional extra is not
    installed. A build that cannot reach OpenAI should still be able to SAY that
    OpenAI exists and that the ``openai`` extra is what it is missing; hiding
    unavailable providers from the catalog would turn a fixable install into an
    unexplained "unknown provider".
    """
    registry = ProviderRegistry()
    registry.register(FAKE_PROVIDER_INFO, build_fake_provider)
    for info, builder in REAL_PROVIDERS:
        registry.register(info, builder)
    return registry


__all__ = [
    "ATTEMPT_METADATA_KEY",
    "CASE_ID_METADATA_KEY",
    "EXPECTED_METADATA_KEY",
    "FAKE_PROVIDER_INFO",
    "REAL_PROVIDERS",
    "RUN_ID_METADATA_KEY",
    "BaseProvider",
    "EnvCredentialResolver",
    "FakeProvider",
    "FakeProviderOptions",
    "ProviderInfo",
    "ProviderRegistry",
    "build_fake_provider",
    "default_registry",
]
