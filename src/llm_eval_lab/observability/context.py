"""Context propagation for structured logs.

``run_id`` and ``case_id`` are bound into structlog's contextvars rather than
threaded through every call signature, so a log line emitted five frames deep
inside an evaluator still says which case it belongs to. contextvars follow
``await`` boundaries and are copied into each task a ``TaskGroup`` spawns,
which is exactly the propagation an async runner needs.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import structlog

RUN_ID_KEY = "run_id"
CASE_ID_KEY = "case_id"


def bind_run(run_id: str) -> None:
    """Bind `run_id` into the ambient logging context."""
    structlog.contextvars.bind_contextvars(**{RUN_ID_KEY: run_id})


def bind_case(case_id: str) -> None:
    """Bind `case_id` into the ambient logging context."""
    structlog.contextvars.bind_contextvars(**{CASE_ID_KEY: case_id})


def clear_context() -> None:
    """Drop every bound value from the ambient logging context."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def case_context(case_id: str) -> Iterator[None]:
    """Bind `case_id` for the duration of the block, then unbind it.

    Unbinding matters under concurrency: a task that finished a case and moved
    on must not keep stamping the previous case's id onto its log lines.
    """
    tokens = structlog.contextvars.bind_contextvars(**{CASE_ID_KEY: case_id})
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
