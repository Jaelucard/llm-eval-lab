"""A per-provider gate that pauses new work after a rate-limit response.

Backing off only the case that received the 429 is not enough: the other
cases in flight keep hitting the same endpoint and keep getting throttled, so
the run makes no progress while generating maximum load. The gate closes for
everybody on the first rate-limit error and reopens once, which is the
difference between backing off and merely retrying.

The gate is an :class:`asyncio.Event` that is normally SET. Closing it clears
the event; a background task re-sets it after the cooldown. Cases waiting on it
resume together.
"""

import asyncio
from types import TracebackType
from typing import Self

DEFAULT_COOLDOWN_S = 1.0
"""Cooldown used when a vendor throttles without saying for how long."""

MAX_COOLDOWN_S = 120.0
"""Ceiling on a vendor-supplied `Retry-After`, so one bad header cannot stall a run."""


class RateLimitGate:
    """An open/closed gate shared by every case targeting one provider."""

    def __init__(self, *, max_cooldown_s: float = MAX_COOLDOWN_S) -> None:
        """Create an open gate."""
        self._open = asyncio.Event()
        self._open.set()
        self._max_cooldown_s = max_cooldown_s
        self._reopen_task: asyncio.Task[None] | None = None
        self._closed_until = 0.0

    @property
    def is_open(self) -> bool:
        """Report whether new work may proceed."""
        return self._open.is_set()

    async def wait(self) -> None:
        """Block until the gate is open. Returns immediately when it already is."""
        await self._open.wait()

    def close_for(self, seconds: float | None) -> float:
        """Close the gate for `seconds`, and return the cooldown actually applied.

        A cooldown already in progress is extended rather than restarted, so two
        simultaneous 429s do not halve the pause the first one asked for.
        """
        cooldown = min(max(seconds or DEFAULT_COOLDOWN_S, 0.0), self._max_cooldown_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cooldown
        if deadline <= self._closed_until:
            return max(self._closed_until - loop.time(), 0.0)

        self._closed_until = deadline
        self._open.clear()
        if self._reopen_task is not None:
            self._reopen_task.cancel()
        self._reopen_task = loop.create_task(self._reopen_after(cooldown))
        return cooldown

    async def _reopen_after(self, cooldown: float) -> None:
        """Reopen the gate after the cooldown elapses.

        A cancellation propagates untouched: the task is cancelled only when a
        longer cooldown has replaced this one, or when the gate is closing down,
        and in neither case should the gate be reopened here.
        """
        await asyncio.sleep(cooldown)
        self._open.set()

    async def aclose(self) -> None:
        """Cancel any pending reopen task and leave the gate open.

        The ``CancelledError`` swallowed here is the REOPEN TASK's, which we just
        requested. If the calling task is itself being cancelled at the same
        moment, that cancellation must survive: ``Task.cancelling()`` is what
        tells the two apart, and without the check a run being shut down could
        have its cancellation quietly absorbed by its own cleanup.
        """
        task = self._reopen_task
        self._reopen_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling() > 0:
                    raise
        self._open.set()

    async def __aenter__(self) -> Self:
        """Enter a scope that closes the gate cleanly on exit."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Cancel any pending reopen task."""
        del exc_type, exc, traceback
        await self.aclose()
