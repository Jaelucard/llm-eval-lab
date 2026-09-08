"""Bounded exponential backoff with jitter. The only retry logic in the project.

Providers perform exactly one attempt per call (decision D4), so this module is
the single place a retry decision is made. That is what makes the recorded
``attempts`` count true: nothing else in the stack is quietly retrying behind
it.

The policy, stated once:

* ``delay = min(max_delay_s, base_delay_s * 2 ** (attempt - 1))``
* full jitter: ``actual = random.uniform(0, delay)``, so a fleet of clients
  that failed together does not retry together
* a ``Retry-After`` from the vendor is honoured as a FLOOR, never a ceiling -
  the server is telling us how long it needs, and backing off less than that is
  just another 429
* stop at ``max_attempts``, or when the cumulative delay would exceed
  ``max_total_delay_s``
* never retry an :class:`asyncio.CancelledError`, which is not a failure but a
  request to stop
"""

import random
from dataclasses import dataclass

from llm_eval_lab.models import ProviderError, RetryPolicy

_DEFAULT_RNG = random.Random()  # noqa: S311 - retry jitter, not cryptography
"""Module-level jitter source. A test injects its own `random.Random` instead."""


@dataclass(frozen=True)
class RetryDecision:
    """Whether to retry, how long to wait, and why not when the answer is no."""

    retry: bool
    delay_s: float = 0.0
    reason: str | None = None


def backoff_delay(policy: RetryPolicy, attempt: int) -> float:
    """Return the un-jittered delay before `attempt` + 1, capped by `max_delay_s`."""
    exponent = max(attempt - 1, 0)
    return min(policy.max_delay_s, policy.base_delay_s * (2.0**exponent))


def apply_jitter(policy: RetryPolicy, delay: float, rng: random.Random) -> float:
    """Apply the configured jitter strategy to a computed delay."""
    if policy.jitter == "none":
        return delay
    if policy.jitter == "equal":
        return delay / 2.0 + rng.uniform(0.0, delay / 2.0)
    return rng.uniform(0.0, delay)


def next_retry(
    policy: RetryPolicy,
    error: ProviderError,
    *,
    attempt: int,
    elapsed_delay_s: float,
    rng: random.Random | None = None,
) -> RetryDecision:
    """Decide whether attempt `attempt` may be retried, and after what delay.

    `attempt` is 1-based and counts the attempt that just FAILED.
    `elapsed_delay_s` is the total time already spent sleeping between attempts
    for this case, which is what bounds a long chain of retries independently of
    the attempt count.
    """
    if not error.retryable:
        return RetryDecision(retry=False, reason=f"{error.kind.value} is not retryable")
    if attempt >= policy.max_attempts:
        return RetryDecision(retry=False, reason=f"exhausted {policy.max_attempts} attempts")

    delay = apply_jitter(policy, backoff_delay(policy, attempt), rng or _DEFAULT_RNG)
    if policy.respect_retry_after and error.retry_after_s is not None:
        delay = max(delay, error.retry_after_s)

    if elapsed_delay_s + delay > policy.max_total_delay_s:
        return RetryDecision(
            retry=False,
            reason=f"retry budget of {policy.max_total_delay_s}s would be exceeded",
        )
    return RetryDecision(retry=True, delay_s=delay)
