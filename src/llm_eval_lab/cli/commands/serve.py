"""``llm-eval serve`` - run the HTTP API and, when one is bundled, the dashboard.

This module owns the FLAGS. What they mean - the ASGI server, its single worker
and why it has to be single - lives in :mod:`llm_eval_lab.api.server`, so
``uvicorn`` keeps exactly one import site in the codebase. The web stack is
imported inside :func:`serve` rather than at module scope, so no other command
pays for it.

**Loopback by default.** The bind address is checked before anything listens: a
non-loopback bind with no ``LLM_EVAL_API_TOKEN`` is refused outright, because
this API starts runs that spend money against the operator's credentials.
"""
# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.

from typing import Annotated

import typer

from llm_eval_lab.cli.output import stderr_console
from llm_eval_lab.reporting.formatters import markup_safe


def serve(
    ctx: typer.Context,
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind address. Loopback unless LLM_EVAL_API_TOKEN is set."),
    ] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port.")] = None,
    access_log: Annotated[
        bool,
        typer.Option("--access-log/--no-access-log", help="Log one line per request."),
    ] = False,
) -> None:
    """Serve the HTTP API on the configured address."""
    # Imported here, not at module scope. `cli/main.py` registers every command at
    # import time, so a module-scope import of `llm_eval_lab.api` would load
    # FastAPI, Starlette and uvicorn on `llm-eval --help` and on every other
    # command - roughly a third of the startup cost of a command that never
    # serves anything. The import contract permits the edge; the cost is what
    # this defers. import-linter reads function-level imports, so the contract is
    # unaffected.
    from llm_eval_lab.api.app import create_app  # noqa: PLC0415 - deferred, see above
    from llm_eval_lab.api.errors import InsecureBindError  # noqa: PLC0415 - deferred
    from llm_eval_lab.api.server import run_server  # noqa: PLC0415 - deferred
    from llm_eval_lab.cli.main import ExitCode, app_context  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    overrides: dict[str, object] = {}
    if host is not None:
        overrides["api_host"] = host
    if port is not None:
        overrides["api_port"] = port
    settings = context.settings.model_copy(update=overrides) if overrides else context.settings

    try:
        application = create_app(settings, bind_host=settings.api_host)
    except InsecureBindError as exc:
        console.print(f"[red]refusing to start:[/red] {markup_safe(exc)}")
        raise typer.Exit(int(ExitCode.USAGE)) from exc

    console.print(
        f"serving on http://{markup_safe(settings.api_host)}:{settings.api_port} "
        f"(api under /api, schema at /openapi.json)"
    )
    try:
        run_server(
            application,
            settings,
            host=settings.api_host,
            port=settings.api_port,
            access_log=access_log,
        )
    except InsecureBindError as exc:  # pragma: no cover - create_app refuses first
        console.print(f"[red]refusing to start:[/red] {markup_safe(exc)}")
        raise typer.Exit(int(ExitCode.USAGE)) from exc


__all__ = ["serve"]
