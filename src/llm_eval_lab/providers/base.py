"""The provider base class and the request-metadata convention.

**One attempt per call.** ``generate`` either returns a successful
:class:`~llm_eval_lab.models.ModelResponse` or raises a
:class:`~llm_eval_lab.models.ProviderError`. It never returns a response
carrying an error, and it never retries: backoff, jitter and rate-limit pacing
belong to the runner (decision D4), so the recorded ``attempts`` count is true
rather than double-counted against an SDK's own retry loop.

**Request metadata is case context, not payload.** The runner stamps a few
neutral keys onto ``ProviderRequest.metadata`` so a provider that simulates
behaviour can see which case and which attempt it is serving. A provider
talking to a real vendor MUST NOT transmit these; they exist for the harness,
not for the model.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from llm_eval_lab.models import ModelResponse, ProviderConfig, ProviderRequest

CASE_ID_METADATA_KEY = "case_id"
"""Metadata key carrying the id of the case being generated."""

RUN_ID_METADATA_KEY = "run_id"
"""Metadata key carrying the id of the run the request belongs to."""

ATTEMPT_METADATA_KEY = "attempt"
"""Metadata key carrying the 1-based attempt number, as a decimal string."""

EXPECTED_METADATA_KEY = "expected"
"""Metadata key carrying the case's expected answer as text, when it has one.

Only the fake provider reads this, to serve its `expected` and `mutate` modes.
A provider that talks to a vendor must never include it in a request body: it
is the answer key.
"""


class BaseProvider(ABC):
    """Shared construction and lifecycle for every provider implementation."""

    name: ClassVar[str]

    def __init__(self, config: ProviderConfig) -> None:
        """Bind the provider to the configuration it will serve."""
        self._config = config

    @property
    def config(self) -> ProviderConfig:
        """Return the configuration this provider was constructed with."""
        return self._config

    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ModelResponse:
        """Perform exactly one generation attempt.

        Returns a successful response, or raises
        :class:`~llm_eval_lab.models.ProviderError`.
        """

    async def aclose(self) -> None:
        """Release transport resources. The default holds none, so this is a no-op."""
        return
