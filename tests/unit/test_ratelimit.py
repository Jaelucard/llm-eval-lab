"""`RateLimitGate`: the shared open/closed gate behind provider rate-limit backoff.

Every test here drives the gate through a fake clock and a controllable stand-in
for `asyncio.sleep`, rather than waiting out real cooldowns. `close_for` reads
the current time from `asyncio.get_running_loop().time()` and schedules its
reopen via `asyncio.sleep`; both are reached through `ratelimit.asyncio`, so
patching that one module attribute to a proxy - real everywhere except those
two entry points - makes every scenario below deterministic and near-instant,
with no dependency on how fast the machine running it happens to be.
"""

import asyncio as real_asyncio
from dataclasses import dataclass, field

import pytest

from llm_eval_lab.runner import ratelimit as ratelimit_module
from llm_eval_lab.runner.ratelimit import DEFAULT_COOLDOWN_S, MAX_COOLDOWN_S, RateLimitGate

YIELD = 0
"""Passed to a REAL `asyncio.sleep` solely to let a scheduled callback run one
tick of the event loop - not a wait on any duration the gate itself owns."""


async def tick() -> None:
    """Yield the event loop once.

    `close_for` is synchronous: it schedules the reopen task via
    `loop.create_task` but returns before that task has run at all, so a test
    asserting on the fake sleep it starts must yield control back to the loop
    first - otherwise `_FakeSleep.__call__` has not executed yet and `calls`
    is still empty. Not a wait on any real duration the gate owns; this always
    resolves on the very next loop iteration.
    """
    await real_asyncio.sleep(YIELD)


@dataclass
class _FakeClock:
    """A monotonic clock `close_for` reads from, advanced only by the test."""

    now: float = 0.0


class _FakeLoop:
    """The real running loop, except `.time()` answers from a `_FakeClock`."""

    def __init__(self, real_loop: real_asyncio.AbstractEventLoop, clock: _FakeClock) -> None:
        self._real_loop = real_loop
        self._clock = clock

    def time(self) -> float:
        """Report the test's own clock, not the wall clock."""
        return self._clock.now

    def __getattr__(self, name: str) -> object:
        """Delegate everything else - `create_task` above all - to the real loop."""
        return getattr(self._real_loop, name)


class _FakeSleep:
    """A controllable stand-in for the `asyncio.sleep` the reopen task awaits.

    Every call is recorded, so a test can assert the exact cooldown `close_for`
    computed, and blocks on a per-instance event the test releases directly -
    never on a real duration. `release()` lets one pending call return and
    arms a fresh event for the next.
    """

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._release = real_asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        await self._release.wait()

    def release(self) -> None:
        """Let one pending fake sleep return, as if its cooldown had elapsed."""
        self._release.set()
        self._release = real_asyncio.Event()


class _FakeAsyncio:
    """Proxies the real `asyncio` module for `ratelimit.py` alone.

    Only `get_running_loop` and `sleep` are overridden; `Event`, `current_task`
    and everything else the module uses reach the real implementation
    unchanged, so the gate's own concurrency primitives behave exactly as they
    do in production.
    """

    def __init__(self, clock: _FakeClock, sleep: _FakeSleep) -> None:
        self._clock = clock
        self._sleep = sleep

    def get_running_loop(self) -> _FakeLoop:
        return _FakeLoop(real_asyncio.get_running_loop(), self._clock)

    async def sleep(self, seconds: float) -> None:
        await self._sleep(seconds)

    def __getattr__(self, name: str) -> object:
        return getattr(real_asyncio, name)


@dataclass
class _Rig:
    """One gate plus the controllable clock and sleep behind it."""

    gate: RateLimitGate
    clock: _FakeClock = field(default_factory=_FakeClock)
    sleep: _FakeSleep = field(default_factory=_FakeSleep)


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    """A gate wired to a fake clock and a fake, test-released sleep."""
    clock = _FakeClock()
    sleep = _FakeSleep()
    monkeypatch.setattr(ratelimit_module, "asyncio", _FakeAsyncio(clock, sleep))
    return _Rig(gate=RateLimitGate(), clock=clock, sleep=sleep)


# -- starting state ----------------------------------------------------------


async def test_a_new_gate_is_open(rig: _Rig) -> None:
    assert rig.gate.is_open is True


async def test_wait_returns_immediately_while_open(rig: _Rig) -> None:
    """An already-set `asyncio.Event` never suspends its waiter, so this
    completing at all (rather than hanging) is the proof; no timeout wrapper
    is used because `wait_for(..., timeout=0)` cancels before an eagerly
    scheduled task gets to run even once, which would make the test wrong for
    the wrong reason."""
    await rig.gate.wait()


# -- close_for: cooldown selection -------------------------------------------


async def test_close_for_closes_the_gate_and_uses_the_requested_cooldown(rig: _Rig) -> None:
    returned = rig.gate.close_for(5.0)
    await tick()

    assert returned == 5.0
    assert rig.gate.is_open is False
    assert rig.sleep.calls == [5.0]


async def test_close_for_falls_back_to_the_default_cooldown_when_seconds_is_none(
    rig: _Rig,
) -> None:
    """A vendor 429 with no `Retry-After` still backs off, for the default window."""
    returned = rig.gate.close_for(None)
    await tick()

    assert returned == DEFAULT_COOLDOWN_S
    assert rig.sleep.calls == [DEFAULT_COOLDOWN_S]


def _huge_retry_after() -> float:
    return MAX_COOLDOWN_S * 100


async def test_close_for_bounds_a_huge_retry_after_at_the_maximum_cooldown(rig: _Rig) -> None:
    """A vendor-supplied `Retry-After` is honoured, but capped, so one bad header
    from one vendor cannot stall the whole run."""
    returned = rig.gate.close_for(_huge_retry_after())
    await tick()

    assert returned == MAX_COOLDOWN_S
    assert rig.sleep.calls == [MAX_COOLDOWN_S]


async def test_close_for_honours_a_specific_retry_after_under_the_cap(rig: _Rig) -> None:
    returned = rig.gate.close_for(3.5)
    await tick()

    assert returned == 3.5
    assert rig.sleep.calls == [3.5]


async def test_close_for_treats_a_negative_retry_after_as_zero(rig: _Rig) -> None:
    """A vendor cannot ask for a cooldown shorter than none at all.

    The clock is moved forward first: at a fresh gate's initial time, a
    zero-length cooldown's deadline coincides with the still-open gate's own
    starting `_closed_until` and is correctly treated as a no-op (see the
    boundary test below) - so proving the FLOOR at zero, rather than that
    coincidence, requires a moment where a zero-length close is still later
    than whatever came before it.
    """
    rig.clock.now = 5.0
    returned = rig.gate.close_for(-10.0)
    await tick()

    assert returned == 0.0
    assert rig.gate.is_open is False
    assert rig.sleep.calls == [0.0]


# -- close_for: overlapping closes --------------------------------------------


async def test_a_second_close_extends_the_cooldown_when_it_would_end_later(rig: _Rig) -> None:
    """Two simultaneous 429s must not halve the pause the first one asked for -
    the later deadline wins and the reopen is rescheduled against it."""
    first = rig.gate.close_for(2.0)
    await tick()
    assert first == 2.0

    rig.clock.now = 1.0
    second = rig.gate.close_for(5.0)
    await tick()

    assert second == 5.0, "the new deadline (1.0 + 5.0 = 6.0) is later, so it applies in full"
    assert rig.gate.is_open is False
    assert rig.sleep.calls == [2.0, 5.0], "the first reopen was cancelled and replaced"


async def test_a_second_close_that_would_end_sooner_is_ignored(rig: _Rig) -> None:
    """A shorter, later 429 must not shrink a cooldown already in progress."""
    first = rig.gate.close_for(5.0)
    await tick()
    assert first == 5.0

    rig.clock.now = 1.0
    second = rig.gate.close_for(0.5)
    await tick()

    assert second == pytest.approx(4.0), "the remaining time on the ORIGINAL cooldown"
    assert rig.gate.is_open is False
    assert rig.sleep.calls == [5.0], "no second reopen was scheduled"


async def test_a_second_close_exactly_at_the_existing_deadline_is_ignored(rig: _Rig) -> None:
    """The boundary is inclusive: a same-length cooldown never replaces the first."""
    rig.gate.close_for(4.0)
    await tick()

    rig.clock.now = 1.0
    second = rig.gate.close_for(3.0)  # 1.0 + 3.0 == 4.0, the existing deadline exactly
    await tick()

    assert second == pytest.approx(3.0)
    assert rig.sleep.calls == [4.0]


# -- reopening -----------------------------------------------------------------


async def test_the_gate_reopens_once_the_cooldown_elapses(rig: _Rig) -> None:
    rig.gate.close_for(1.0)
    await tick()
    assert rig.gate.is_open is False

    rig.sleep.release()
    await tick()

    assert rig.gate.is_open is True


async def test_concurrent_waiters_all_pause_and_all_resume_together(rig: _Rig) -> None:
    """The gate closes for EVERY case in flight, not only the one that hit the 429,
    and reopens them all in the same moment rather than one at a time."""
    rig.gate.close_for(1.0)

    waiters = [real_asyncio.ensure_future(rig.gate.wait()) for _ in range(5)]
    await tick()
    assert all(not waiter.done() for waiter in waiters), "closed: nobody may proceed yet"

    rig.sleep.release()
    # One tick lets the reopen task wake up and call `self._open.set()`; a
    # second lets `asyncio.Event.set()`'s own callbacks actually resume and
    # complete every waiter it woke.
    await tick()
    await tick()

    assert all(waiter.done() for waiter in waiters), "reopened: everybody proceeds together"


# -- aclose --------------------------------------------------------------------


async def test_aclose_with_nothing_pending_leaves_the_gate_open(rig: _Rig) -> None:
    await rig.gate.aclose()
    assert rig.gate.is_open is True


async def test_aclose_cancels_a_pending_reopen_and_leaves_the_gate_open(rig: _Rig) -> None:
    """The ordinary shutdown path: cancel the stale reopen task, then force-open."""
    rig.gate.close_for(30.0)
    assert rig.gate.is_open is False

    await rig.gate.aclose()

    assert rig.gate.is_open is True


async def test_the_async_context_manager_calls_aclose_on_exit(rig: _Rig) -> None:
    async with rig.gate as gate:
        gate.close_for(30.0)
        assert gate.is_open is False

    assert rig.gate.is_open is True


async def test_aclose_does_not_swallow_the_callers_own_cancellation(rig: _Rig) -> None:
    """`aclose()`'s own `except CancelledError` catches the REOPEN TASK'S
    cancellation, which it just requested. If the CALLER is itself being
    cancelled at that exact moment - the run is shutting down - that
    cancellation must propagate rather than be absorbed by this cleanup, and
    the gate must not be force-opened on that path: the surrounding shutdown
    is what decides its final state, not this cleanup.
    """
    rig.gate.close_for(30.0)

    async def _caller() -> None:
        await rig.gate.aclose()

    outer = real_asyncio.ensure_future(_caller())
    # Let `_caller` run up to `await task` inside `aclose`, where it is
    # suspended waiting for the (already-cancelled) reopen task to finish.
    await real_asyncio.sleep(YIELD)

    outer.cancel()
    with pytest.raises(real_asyncio.CancelledError):
        await outer

    assert rig.gate.is_open is False, "the caller's cancellation pre-empted the force-open"
