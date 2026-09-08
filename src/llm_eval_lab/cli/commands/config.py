# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval config show`` - the effective configuration, with no secrets in it.

**A credential is rendered as its environment variable NAME plus a set or unset
boolean. Never as a value, and never as a partially masked value.** A mask is
still a disclosure: it leaks the length, the prefix and often enough of the
tail to identify the key, and it invites somebody to widen it "just a little"
later. There is nothing to widen here, because no code path in this module
reads a credential value.

**A credential also hides inside a URL.** ``database_url`` is a plain string on
``Settings``, and a PostgreSQL DSN carries its password in the userinfo, so
printing the effective configuration verbatim published it. Every value goes
through :func:`~llm_eval_lab.redaction.redact_setting_value`, which is the one
implementation of that rule and lives beside the logging and persistence
scrubbers rather than here, so a future API configuration endpoint cannot
re-derive a weaker version of it.

Precedence is ``CLI flags > environment > config file > defaults``, and it is
reported per field rather than only as a heading. The environment layer is
identified from ``Settings.model_fields_set`` rather than by reading
``os.environ``, which ``settings.py`` alone is permitted to touch (decision D8).
"""

from typing import Annotated, Any

import typer
from pydantic import SecretStr

from llm_eval_lab.cli.output import emit_json, key_value_table, stderr_console
from llm_eval_lab.redaction import redact_setting_value, redact_url
from llm_eval_lab.services import CatalogService, build_catalog_service

app = typer.Typer(
    name="config",
    help="Inspect the effective configuration.",
    no_args_is_help=True,
)

ENV_PREFIX = "LLM_EVAL_"
"""Every settings field is read from this prefix plus its upper-cased name."""

SOURCE_CLI = "cli"
SOURCE_ENVIRONMENT = "environment"
SOURCE_CONFIG_FILE = "config-file"
SOURCE_DEFAULT = "default"

PRECEDENCE = (SOURCE_CLI, SOURCE_ENVIRONMENT, SOURCE_CONFIG_FILE, SOURCE_DEFAULT)
"""The resolution order, highest first. `resolve_settings` implements exactly this."""


def env_var_for(field: str) -> str:
    """Return the environment variable name a settings field is read from."""
    return f"{ENV_PREFIX}{field.upper()}"


def source_of(
    name: str,
    *,
    cli_overrides: dict[str, Any],
    env_provided: frozenset[str],
    file_overrides: dict[str, Any],
) -> str:
    """Return which layer supplied one field's effective value.

    Checked in the documented precedence order, which is the same order
    ``resolve_settings`` applies, so this reports what happened rather than what
    was intended.
    """
    if name in cli_overrides:
        return SOURCE_CLI
    if name in env_provided:
        return SOURCE_ENVIRONMENT
    if name in file_overrides:
        return SOURCE_CONFIG_FILE
    return SOURCE_DEFAULT


def config_document(
    values: dict[str, Any],
    *,
    cli_overrides: dict[str, Any],
    env_provided: frozenset[str],
    file_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one row per settings field: its value, its variable name and its source.

    Every value passes through the shared redaction rule, so a password inside a
    connection string is replaced before it can reach either output form.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        is_secret = isinstance(value, SecretStr)
        rows.append(
            {
                "setting": name,
                "value": redact_setting_value(value),
                "is_secret": is_secret,
                "set": value is not None if is_secret else None,
                "env_var": env_var_for(name),
                "source": source_of(
                    name,
                    cli_overrides=cli_overrides,
                    env_provided=env_provided,
                    file_overrides=file_overrides,
                ),
            }
        )
    return rows


def credential_document(catalog: CatalogService) -> list[dict[str, Any]]:
    """List every provider credential as a variable NAME plus whether it is set."""
    return [
        {
            "provider": info.name,
            "env_var": info.credential_env,
            "required": info.requires_credential,
            "set": catalog.credential_status(info.name),
        }
        for info in catalog.providers()
        if info.requires_credential
    ]


def _describe(row: dict[str, Any]) -> str:
    """Render one settings row for the human table, never unwrapping a secret."""
    if row["is_secret"]:
        return f"{row['env_var']}: {'set' if row['set'] else 'not set'}"
    value = row["value"]
    return "-" if value is None else str(value)


@app.command("show")
def show(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Print the effective configuration with precedence resolved."""
    from llm_eval_lab.cli.main import app_context, guard  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        settings = context.settings
        catalog = build_catalog_service(settings)
        rows = config_document(
            {name: getattr(settings, name) for name in type(settings).model_fields},
            cli_overrides=context.cli_overrides,
            env_provided=context.env_provided,
            file_overrides=context.file_overrides,
        )
        credentials = credential_document(catalog)
        database_url = redact_url(context.database_url)

        if as_json:
            emit_json(
                "config show",
                {
                    "ok": True,
                    "precedence": list(PRECEDENCE),
                    "config_file": str(context.config_path) if context.config_path else None,
                    "database_url": database_url,
                    "settings": rows,
                    "credentials": credentials,
                },
            )
            return

        console.print(
            key_value_table(
                "effective configuration",
                {
                    "precedence": " > ".join(PRECEDENCE),
                    "config file": str(context.config_path) if context.config_path else None,
                    "database url": database_url,
                },
            )
        )
        console.print(
            key_value_table(
                "settings",
                {f"{row['setting']} [{row['source']}]": _describe(row) for row in rows},
            )
        )
        console.print(
            key_value_table(
                "provider credentials (variable names only)",
                {
                    str(entry["provider"]): (
                        f"{entry['env_var']}: {'set' if entry['set'] else 'not set'}"
                    )
                    for entry in credentials
                }
                or {"(none)": "no registered provider requires a credential"},
            )
        )

    guard(context, "config show", as_json=as_json, action=action)


__all__ = ["app", "config_document", "credential_document", "env_var_for", "show", "source_of"]
