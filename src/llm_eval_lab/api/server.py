"""How this API is served. Owned by `api`, so no other package imports uvicorn.

The CLI owns the flags; this module owns what those flags mean. Keeping the
ASGI server here means ``uvicorn`` has exactly one import site in the codebase,
which is what lets the ``fastapi, uvicorn and starlette are api-only`` contract
in `pyproject.toml` stay true while ``llm-eval serve`` still exists.

**This module is where the bind is enforced.** `create_app` checks the address
it is TOLD about, which is the best it can do: any ASGI entry point may bind
something it never mentioned. Running
``uvicorn --factory llm_eval_lab.api.app:create_app --host 0.0.0.0`` bypasses the
CLI entirely, and the factory has no way to learn that the socket faces the
network. :func:`build_server` is handed the real address and re-checks it there,
so every path that goes through ``llm-eval serve`` is safe by construction. A
deployment that starts the application some other way is responsible for setting
``LLM_EVAL_API_TOKEN``, and it is called out here rather than left implicit.

**One worker, and the reason is not performance.** A run launched through the
API is an ``asyncio`` task in the memory of the process that launched it,
together with its cancel event (decision D-RUN). A second worker would answer
``POST /api/runs/{id}/cancel`` for a run it has never heard of, and would answer
it with a ``202`` that nothing acts on. ``workers=1`` is therefore passed
explicitly, next to this note, rather than left to a default that could change.
"""

import uvicorn
from fastapi import FastAPI

from llm_eval_lab.api.app import require_loopback_or_token
from llm_eval_lab.settings import Settings


def build_server(
    app: FastAPI,
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    access_log: bool = False,
) -> uvicorn.Server:
    """Configure the ASGI server for `app` without starting it.

    The bind check runs here, against the address this server will ACTUALLY
    listen on, which is the point: `create_app` can only check what it was told,
    and it is not told when a caller overrides the host afterwards. Separate from
    :func:`run_server` so a caller - a test, or a future supervisor - can inspect
    the configuration, and so the refusal is observable without starting a socket.

    Raises:
        InsecureBindError: when the bind would expose an unauthenticated API.
    """
    bind_host = settings.api_host if host is None else host
    bind_port = settings.api_port if port is None else port
    require_loopback_or_token(settings, bind_host)
    return uvicorn.Server(
        uvicorn.Config(
            app,
            host=bind_host,
            port=bind_port,
            # See the module docstring: a second worker cannot see the runs the
            # first is executing, so status and cancellation would both lie.
            workers=1,
            log_level=settings.log_level.lower(),
            access_log=access_log,
        )
    )


def run_server(
    app: FastAPI,
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    access_log: bool = False,
) -> None:
    """Serve `app` on the given address until the process is interrupted.

    Raises:
        InsecureBindError: when the bind would expose an unauthenticated API.
    """
    build_server(app, settings, host=host, port=port, access_log=access_log).run()


__all__ = ["build_server", "run_server"]
