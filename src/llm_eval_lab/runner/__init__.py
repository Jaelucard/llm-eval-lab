"""The async run engine: concurrency, retries, rate limiting and progress.

Everything about how a run behaves under partial failure is decided in
``engine.py``; ``retry.py`` owns backoff, ``ratelimit.py`` owns per-provider
pacing, and ``progress.py`` emits the observation events the CLI and the API
render.
"""

from llm_eval_lab.runner.engine import (
    BUDGET_EXCEEDED,
    MAX_ERROR_RATE_EXCEEDED,
    ProviderFactory,
    RunEngine,
    RunOutcome,
    aggregate_case_verdict,
)
from llm_eval_lab.runner.progress import ProgressEmitter, null_callback
from llm_eval_lab.runner.ratelimit import RateLimitGate
from llm_eval_lab.runner.retry import RetryDecision, backoff_delay, next_retry

__all__ = [
    "BUDGET_EXCEEDED",
    "MAX_ERROR_RATE_EXCEEDED",
    "ProgressEmitter",
    "ProviderFactory",
    "RateLimitGate",
    "RetryDecision",
    "RunEngine",
    "RunOutcome",
    "aggregate_case_verdict",
    "backoff_delay",
    "next_retry",
    "null_callback",
]
