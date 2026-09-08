"""`RunManager`'s reservation bookkeeping: `in_flight()` and shutdown's early return.

These two facts are tested in isolation from a real run, because both are
about the ACCOUNTING around `_tracked` and `_reserved`, not about anything a
run's execution touches. `_reserve` is called directly, the way `launch`
would between claiming a slot and a task existing to track - the exact window
`shutdown`'s early-return bug lived in.
"""

from typing import TYPE_CHECKING, cast

import structlog

from llm_eval_lab.services.run_manager import RunManager

if TYPE_CHECKING:
    from llm_eval_lab.services.catalog_service import CatalogService
    from llm_eval_lab.services.run_service import RunService


def _manager() -> RunManager:
    """A `RunManager` with no real collaborators.

    Safe because neither test below reaches `launch`, `cancel` or `status` -
    the only methods that would actually call into `runs` or `catalog`.
    """
    logger: structlog.BoundLogger = structlog.get_logger("tests")
    return RunManager(
        runs=cast("RunService", object()),
        catalog=cast("CatalogService", object()),
        logger=logger,
    )


def test_in_flight_counts_tracked_and_reserved_runs() -> None:
    manager = _manager()
    assert manager.in_flight() == 0

    first = manager._reserve()
    assert manager.in_flight() == 1

    manager._reserved.discard(first)
    manager._tracked["run-1"] = object()  # type: ignore[assignment]
    assert manager.in_flight() == 1

    manager._reserve()
    assert manager.in_flight() == 2, "one tracked run plus one reserved slot"


async def test_shutdown_clears_a_reservation_even_with_no_tracked_task() -> None:
    """The early-return path: a reservation with nothing yet tracked for it.

    Before this, `shutdown` computed `_pending_tasks()` from `_tracked` alone,
    found nothing, and returned before `_reserved` was ever touched - leaking
    the slot for the rest of the process's life every time shutdown raced a
    launch between `_reserve()` and the task being tracked.
    """
    manager = _manager()
    manager._reserve()
    assert manager.in_flight() == 1

    await manager.shutdown()

    assert manager.in_flight() == 0
    assert not manager._reserved
