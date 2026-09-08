"""The one mechanism this package has for running untrusted work under a clock.

**Why a separate process, and not a worker thread.** CPython runs a regular
expression match inside a single C call that never returns to the interpreter
loop and never releases the GIL. A catastrophic pattern running in a thread
therefore does not merely outlive its timeout - it starves every other thread in
the process, including the event loop that was supposed to time it out, so the
timeout never fires and the whole run hangs. That was measured, not assumed.
Killing a process is the only mechanism CPython offers that actually stops such
a call in progress.

**Why this module exists at all.** Two evaluators run benchmark-supplied regular
expressions against model output: the ``regex`` evaluator directly, and
``json_schema`` through JSON Schema's ``pattern`` and ``patternProperties``
keywords. A second, separately-written guard for the second case would be a
security control with two implementations, which is a control with one
implementation and one bug. Both go through :func:`run_worker`.

**Interpreter flags.** ``-I`` always: isolated mode ignores ``PYTHONPATH`` and
the user site directory, so the child cannot be steered by the environment.
``-S`` additionally skips ``site``, which leaves the child with the standard
library alone and costs roughly 15 ms to start. A worker that needs an installed
third-party package (``jsonschema``) cannot use ``-S`` and pays a larger startup,
which is why ``stdlib_only`` is a parameter rather than a constant: each worker
takes the cheapest isolation that still lets it run.
"""

import asyncio
import sys

from llm_eval_lab.models import EvaluatorError

MAX_WORKER_STDERR_CHARS = 2000
"""Cap on the child's error output quoted back into an exception message."""


class WorkerError(EvaluatorError):
    """A bounded worker process could not produce a result.

    Distinct from a timeout, which surfaces as :class:`TimeoutError` from
    :func:`run_worker` so a caller can report the two differently: a timeout
    means the work was abandoned, this means it never ran.
    """


async def run_worker(
    source: str,
    payload: bytes,
    *,
    timeout_s: float,
    stdlib_only: bool = True,
) -> bytes:
    """Run `source` in a fresh interpreter, feed it `payload`, return its stdout.

    The child is started with no inherited file descriptors beyond the three
    pipes, no project imports, and no access to this process's memory. On expiry
    it is killed and reaped before the timeout propagates, so a runaway match
    cannot outlive the case that started it.

    Args:
        source: The complete program the child runs, passed with ``-c``. It must
            read its request from stdin and write JSON to stdout.
        payload: The request, already encoded. Callers cap its size themselves,
            because what "too big" means differs per worker.
        timeout_s: Wall-clock bound on the whole child lifetime, startup
            included.
        stdlib_only: Start the child with ``-S`` as well as ``-I``, excluding
            site-packages. True for a worker that imports only the standard
            library; False for one that needs an installed distribution.

    Returns:
        The child's stdout, verbatim.

    Raises:
        TimeoutError: when the child did not finish within `timeout_s`. It is
            killed and reaped first.
        WorkerError: when the child exited non-zero or wrote nothing.
        asyncio.CancelledError: when the caller was cancelled; the child is
            killed and reaped first, exactly as for a timeout.
    """
    flags = ["-I", "-S"] if stdlib_only else ["-I"]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *flags,
        "-c",
        source,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout_s):
            stdout, stderr = await process.communicate(payload)
    except (TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.wait()
        raise

    if process.returncode != 0 or not stdout:
        detail = stderr.decode("utf-8", errors="replace").strip() or "no output"
        msg = f"worker exited with {process.returncode}: {detail[:MAX_WORKER_STDERR_CHARS]}"
        raise WorkerError(msg)
    return stdout


__all__ = ["MAX_WORKER_STDERR_CHARS", "WorkerError", "run_worker"]
