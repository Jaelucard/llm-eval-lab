"""The progress emitter. The event TYPES live in the frozen contract layer.

``ProgressEventType``, ``ProgressEvent`` and ``ProgressCallback`` are defined in
``llm_eval_lab.models`` because three lanes consume them: the runner emits, the
CLI renders a progress bar, and the API's run manager snapshots them for the run
status endpoint. This module owns only the counter bookkeeping and the event
construction.

The callback is SYNCHRONOUS and must not block: it is invoked from inside the
run's ``TaskGroup``, so a callback that awaits I/O stalls the run. A consumer
needing I/O queues the event and returns.

A callback must also not RAISE, and nothing here catches it. The project permits
exactly one broad exception boundary, the runner's per-case isolation, and
adding a second one for progress reporting would be the start of exactly the
error-swallowing this codebase forbids. A callback raising from inside a case is
recorded by that boundary like any other failure.
"""

from llm_eval_lab.models import ProgressCallback, ProgressEvent, ProgressEventType
from llm_eval_lab.utils.time import utc_now


def null_callback(event: ProgressEvent) -> None:
    """Discard a progress event. The default when no consumer is interested."""
    del event


class ProgressEmitter:
    """Maintains the run's counters and emits one well-formed event per transition.

    ``total`` is set once at construction and stamped on EVERY event, including
    the first, so a consumer can render "0 of 6" before any case completes
    rather than "0 of 0".

    The counters need no lock. Every task in the run's ``TaskGroup`` shares one
    event loop and one thread, so an increment cannot be interleaved: the only
    suspension points are ``await``s, and there are none between reading a
    counter and writing it. A lock here would suggest a threading model this
    project does not have.
    """

    def __init__(self, run_id: str, total: int, callback: ProgressCallback | None = None) -> None:
        """Create an emitter for one run."""
        self.run_id = run_id
        self.total = total
        self._callback = callback or null_callback
        self._completed = 0
        self._passed = 0
        self._errors = 0

    @property
    def completed(self) -> int:
        """Return how many cases have reached a terminal state."""
        return self._completed

    def _emit(
        self,
        event_type: ProgressEventType,
        *,
        case_id: str | None = None,
        attempt: int | None = None,
        retry_in_s: float | None = None,
        message: str | None = None,
    ) -> None:
        """Build and deliver one event. The callback must neither block nor raise."""
        event = ProgressEvent(
            type=event_type,
            run_id=self.run_id,
            case_id=case_id,
            completed=self._completed,
            total=self.total,
            passed=self._passed,
            errors=self._errors,
            attempt=attempt,
            retry_in_s=retry_in_s,
            message=message,
            at=utc_now(),
        )
        self._callback(event)

    def run_started(self) -> None:
        """Emit the single RUN_STARTED event."""
        self._emit(ProgressEventType.RUN_STARTED)

    def case_started(self, case_id: str) -> None:
        """Emit CASE_STARTED for one case."""
        self._emit(ProgressEventType.CASE_STARTED, case_id=case_id)

    def case_retry(self, case_id: str, *, attempt: int, retry_in_s: float, reason: str) -> None:
        """Emit CASE_RETRY, carrying the attempt number and the pending delay."""
        self._emit(
            ProgressEventType.CASE_RETRY,
            case_id=case_id,
            attempt=attempt,
            retry_in_s=retry_in_s,
            message=reason,
        )

    def evaluation_completed(self, case_id: str, evaluator_id: str) -> None:
        """Emit EVALUATION_COMPLETED for one evaluator on one case."""
        self._emit(ProgressEventType.EVALUATION_COMPLETED, case_id=case_id, message=evaluator_id)

    def case_completed(self, case_id: str, *, passed: bool | None) -> None:
        """Emit the terminal CASE_COMPLETED event and advance the counters."""
        self._completed += 1
        if passed:
            self._passed += 1
        self._emit(ProgressEventType.CASE_COMPLETED, case_id=case_id)

    def case_failed(self, case_id: str, *, message: str | None = None) -> None:
        """Emit the terminal CASE_FAILED event and advance the counters."""
        self._completed += 1
        self._errors += 1
        self._emit(ProgressEventType.CASE_FAILED, case_id=case_id, message=message)

    def run_completed(self, *, message: str | None = None) -> None:
        """Emit the single RUN_COMPLETED event."""
        self._emit(ProgressEventType.RUN_COMPLETED, message=message)
