# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
# FBT is a good rule for library APIs and simply does not apply to a command
# signature that a CLI framework reflects over.
"""The Typer application, the global options and the single source of exit codes.

Exit codes are defined once, here, because a CI pipeline keys off them and a
second definition somewhere else would eventually disagree:

===== ==============================================================
0     success
1     internal error
2     usage error
3     benchmark validation failed
4     regression detected
5     run incomplete
6     runs are incomparable
7     regression warning
130   interrupted
===== ==============================================================

Codes 4, 6 and 7 belong to the comparison commands added in a later slice; they
are listed here now so there is one table rather than two.
"""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.markup import escape

from llm_eval_lab import __version__
from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.models import (
    BenchmarkValidationError,
    LLMEvalError,
    PluginRecord,
)
from llm_eval_lab.observability.logging import configure_logging, get_logger
from llm_eval_lab.settings import Settings, get_settings


class ExitCode(IntEnum):
    """Every exit code this CLI can produce. The single source of truth."""

    SUCCESS = 0
    INTERNAL_ERROR = 1
    USAGE = 2
    VALIDATION_FAILED = 3
    REGRESSION = 4
    RUN_INCOMPLETE = 5
    INCOMPARABLE = 6
    REGRESSION_WARNING = 7
    INTERRUPTED = 130


@dataclass
class AppContext:
    """Resolved global options, handed to every command through the Typer context.

    The three override layers are kept apart rather than merged so
    ``llm-eval config show`` can say which one supplied each value.
    ``env_provided`` names the fields the environment actually set, read off
    ``Settings.model_fields_set`` rather than by inspecting ``os.environ``,
    which ``settings.py`` alone is permitted to touch (decision D8).
    """

    settings: Settings
    database_url: str
    color: bool
    cli_overrides: dict[str, Any] = field(default_factory=dict)
    file_overrides: dict[str, Any] = field(default_factory=dict)
    env_provided: frozenset[str] = frozenset()
    config_path: Path | None = None
    plugins: tuple[PluginRecord, ...] = ()
    """Evaluator plugins loaded at process start. See `load_evaluator_plugins`."""


def load_config_overrides(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON settings-override file.

    Raises:
        typer.BadParameter: when the file is missing or is not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read config file {path}: {exc.strerror or exc}"
        raise typer.BadParameter(msg) from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"config file {path} is not valid YAML or JSON: {exc}"
        raise typer.BadParameter(msg) from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        msg = f"config file {path} must contain a mapping of settings"
        raise typer.BadParameter(msg)
    return document


def environment_provided_fields() -> frozenset[str]:
    """Return the settings fields the environment (or a `.env` file) actually supplied.

    Read off ``Settings.model_fields_set`` on an instance built with no init
    arguments, so the CLI learns which variables were set without reading
    ``os.environ`` itself. ``settings.py`` remains the only module that touches
    the environment (decision D8).
    """
    return frozenset(get_settings().model_fields_set)


def resolve_settings(
    cli_overrides: dict[str, Any],
    file_overrides: dict[str, Any],
) -> Settings:
    """Resolve settings in the documented order: CLI > environment > file > defaults.

    Pydantic-settings ranks init keyword arguments ABOVE environment variables,
    so passing the config file through as init arguments would silently invert
    two of those four layers. Only the CLI flags are passed that way. A config
    file value is applied only for a field the environment did not set, which
    puts the file below the environment and above the defaults exactly as
    ``llm-eval config show`` reports.
    """
    from_environment = environment_provided_fields()
    applied_file = {
        name: value for name, value in file_overrides.items() if name not in from_environment
    }
    overrides = {**applied_file, **cli_overrides}
    return Settings(**overrides) if overrides else get_settings()


def load_evaluator_plugins(settings: Settings) -> tuple[PluginRecord, ...]:
    """Register every allowlisted evaluator plugin, once, at process start.

    The ONE call site this project makes to
    :func:`~llm_eval_lab.evaluators.plugins.load_default_plugins` (decision
    D-CUSTOM). ``api/`` never calls it - a plugin is arbitrary third-party
    code, and the HTTP surface can be reached by anything that can reach the
    port - and neither does ``services/``, which both the CLI and the API
    share: ``tests/unit/test_plugins.py`` asserts by static analysis that
    neither package so much as mentions plugin loading, which is a stronger
    guarantee than patching one call site and hoping nothing else grew a
    second.

    Imported here as a deferred, function-local import rather than at module
    scope, so ``llm-eval --help`` does not pay for loading the evaluator
    registries. ``pyproject.toml``'s import-linter contract carries a single,
    named exemption for exactly this edge (``cli.main ->
    evaluators.plugins``); nothing else under ``cli`` may import
    ``evaluators`` directly.

    Returns:
        The provenance of what was actually loaded, threaded by the caller
        into ``RunConfig.plugins``. Empty when ``settings.plugins_enabled`` is
        false or ``settings.plugins_allowed`` is empty - both are the default.

    Raises:
        EvaluatorConfigError: when an allowlisted entry point cannot be
            imported, does not resolve to evaluator classes, or collides with
            an already-registered type. Aborting startup is deliberate: a run
            that silently lost an evaluator would report a pass rate over the
            wrong set of checks.
    """
    from llm_eval_lab.evaluators.plugins import load_default_plugins  # noqa: PLC0415 - see above

    return load_default_plugins(
        enabled=settings.plugins_enabled,
        allowed=settings.plugins_allowed,
    )


def unloaded_evaluator_plugins(
    settings: Settings, records: tuple[PluginRecord, ...]
) -> tuple[str, ...]:
    """Return allowlisted plugin names that `load_evaluator_plugins` did not load.

    An allowlist entry with nothing behind it is almost always a typo or a
    package that failed to install, so the caller logs these as a warning.
    """
    from llm_eval_lab.evaluators.plugins import unloaded_names  # noqa: PLC0415 - see above

    return unloaded_names(settings.plugins_allowed, records)


app = typer.Typer(
    name="llm-eval",
    help="A local-first LLM evaluation platform: benchmarks, providers and statistics.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.callback()
def main(  # noqa: PLR0913, PLR0917 - one parameter per documented global option
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="YAML or JSON file of settings overrides."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL for this invocation only."),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR or CRITICAL."),
    ] = None,
    log_format: Annotated[
        str | None,
        typer.Option("--log-format", help="console or json."),
    ] = None,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colour in human-facing output."),
    ] = False,
) -> None:
    """Resolve the global options and configure logging for this process."""
    file_overrides: dict[str, Any] = load_config_overrides(config) if config is not None else {}
    cli_overrides: dict[str, Any] = {}
    if log_level is not None:
        cli_overrides["log_level"] = log_level.upper()
    if log_format is not None:
        cli_overrides["log_format"] = log_format
    if db_url is not None:
        cli_overrides["database_url"] = db_url

    settings = resolve_settings(cli_overrides, file_overrides)
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        log_prompts=settings.log_prompts,
        colors=not no_color,
    )
    plugin_records = load_evaluator_plugins(settings)
    for name in unloaded_evaluator_plugins(settings, plugin_records):
        logger.warning("plugin_not_advertised", entry_point=name)
    ctx.obj = AppContext(
        settings=settings,
        database_url=settings.resolved_database_url(),
        color=not no_color,
        cli_overrides=cli_overrides,
        file_overrides=file_overrides,
        env_provided=environment_provided_fields(),
        config_path=config,
        plugins=plugin_records,
    )


@app.command()
def version(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Print the installed version."""
    context = app_context(ctx)
    if as_json:
        emit_json("version", {"version": __version__})
        return
    stderr_console(color=context.color).print(__version__)


def app_context(ctx: typer.Context) -> AppContext:
    """Return the resolved global options for the current invocation."""
    obj = ctx.obj
    if isinstance(obj, AppContext):
        return obj
    # Reached only when a command is invoked without the Typer callback having
    # run, which happens in unit tests that call a command function directly.
    settings = get_settings()
    return AppContext(settings=settings, database_url=settings.resolved_database_url(), color=True)


def run_async[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run one coroutine to completion. The single asyncio entry point in the CLI."""
    return asyncio.run(coroutine)


def guard[T](
    context: AppContext,
    command: str,
    *,
    as_json: bool,
    action: Callable[[], T],
) -> T:
    """Run `action`, mapping the project's exceptions onto the exit-code table.

    Bounded to :class:`~llm_eval_lab.models.LLMEvalError` and
    ``KeyboardInterrupt``. Anything else propagates: an unexpected exception is
    a bug, and hiding it behind exit code 1 with no traceback would make it
    unfixable.
    """
    console = stderr_console(color=context.color)
    try:
        return action()
    except BenchmarkValidationError as exc:
        if as_json:
            emit_json(
                command,
                {
                    "ok": False,
                    "error": "validation_failed",
                    "message": str(exc),
                    "errors": [
                        {
                            "file": item.file,
                            "location": item.location,
                            "case_id": item.case_id,
                            "message": item.message,
                            "input": item.input_repr,
                        }
                        for item in exc.errors
                    ],
                },
            )
        else:
            # `escape`, because a validation message contains `[case has-neither]`
            # and Rich would parse that as a style tag and swallow the case id -
            # the one piece of information the message exists to carry. The same
            # path renders `input_repr` straight from the user's file, so this is
            # output integrity as much as it is a missing id.
            console.print(f"[red]validation failed:[/red] {escape(str(exc))}")
        raise typer.Exit(int(ExitCode.VALIDATION_FAILED)) from exc
    except LLMEvalError as exc:
        if as_json:
            emit_json(
                command,
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
            )
        else:
            console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(int(ExitCode.INTERNAL_ERROR)) from exc
    except KeyboardInterrupt as exc:
        console.print("[yellow]interrupted[/yellow]")
        raise typer.Exit(int(ExitCode.INTERRUPTED)) from exc


logger = get_logger("llm_eval_lab.cli")
"""Module-level logger; commands bind run-scoped context onto it."""


def register_commands() -> None:
    """Attach every command module to the application.

    Imported inside a function so that ``llm-eval --help`` does not pay for
    loading the storage and runner stacks, and so the command modules can import
    ``main`` for the exit-code table without a circular import at module scope.
    """
    from llm_eval_lab.cli.commands import (  # noqa: PLC0415 - deferred by design, see above
        compare,
        db,
        evaluators,
        export,
        metrics,
        models,
        pricing,
        providers,
        report,
        run,
        runs,
        serve,
        show,
        validate,
    )
    from llm_eval_lab.cli.commands import (  # noqa: PLC0415 - deferred by design, see above
        config as config_command,
    )

    app.add_typer(db.app, name="db")
    app.add_typer(pricing.app, name="pricing")
    app.add_typer(config_command.app, name="config")
    app.command(name="validate")(validate.validate)
    app.command(name="run")(run.run)
    app.command(name="runs")(runs.runs)
    app.command(name="show")(show.show)
    app.command(name="metrics")(metrics.metrics)
    app.command(name="compare")(compare.compare)
    app.command(name="report")(report.report)
    app.command(name="export")(export.export)
    app.command(name="models")(models.models)
    app.command(name="providers")(providers.providers)
    app.command(name="evaluators")(evaluators.evaluators)
    app.command(name="serve")(serve.serve)


register_commands()


__all__ = [
    "AppContext",
    "ExitCode",
    "app",
    "app_context",
    "guard",
    "load_evaluator_plugins",
    "run_async",
    "unloaded_evaluator_plugins",
]
