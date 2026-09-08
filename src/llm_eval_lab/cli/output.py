"""The output convention: JSON on stdout, everything human on stderr.

The rule is absolute and it is what makes ``llm-eval run ... --json | jq``
work. With ``--json``, stdout carries exactly one JSON document and nothing
else. Every progress bar, table, warning and log line goes to stderr,
unconditionally, whether or not ``--json`` was passed.

Every JSON document carries a ``schema_version``, so a consumer can tell a
format change from a data change.
"""

import json
import sys
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any

from rich.console import Console
from rich.table import Table

from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.utils.time import isoformat_z

SCHEMA_VERSION = 1
"""Version of the CLI's JSON envelope. Bumped only for a breaking shape change."""


def stderr_console(*, color: bool = True) -> Console:
    """Return the console every human-facing message is written to."""
    return Console(stderr=True, no_color=not color, soft_wrap=True)


def json_default(value: object) -> str:
    """Encode the non-JSON types the command payloads carry.

    Typed rather than a blanket ``str``, because ``str(datetime)`` produces
    ``"2026-09-06 10:40:06.626690+00:00"`` - a SPACE where RFC 3339 requires a
    ``T``. Python's own ``fromisoformat`` accepts that; browsers and most
    non-Python parsers do not, and this envelope is what the HTTP API and the
    dashboard consume.

    ``Decimal`` becomes its exact textual form rather than a float, which is the
    same reason the ``Money`` column type exists: a cost of
    ``0.0000012345`` must survive the trip unchanged.

    The UTC offset is rendered as a trailing ``Z``. Pydantic already emits
    ``Z`` for every timestamp nested inside a dumped model, so without this the
    same envelope carried two spellings of the same instant - ``...Z`` for a
    nested field and ``...+00:00`` for a top-level one - and a consumer
    comparing them as strings would find them unequal.

    Raises:
        TypeError: for anything else, so a new unencodable type is a loud
            failure here rather than a silent ``repr`` in a consumer's data.
    """
    if isinstance(value, datetime):
        # Aware timestamps are a contract invariant; a naive one would serialize
        # without an offset and silently change meaning, so `isoformat_z`
        # stamps UTC before rendering.
        return isoformat_z(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, PurePath):
        return str(value)
    msg = f"{type(value).__name__} is not JSON-serializable in the CLI envelope"
    raise TypeError(msg)


def json_document(command: str, payload: dict[str, Any]) -> str:
    """Render one command's envelope as a JSON string.

    Separate from :func:`emit_json` so a command that writes to a file uses the
    same encoder as one that writes to stdout, rather than reaching for a second
    ``json.dumps`` with different settings.
    """
    document = {"schema_version": SCHEMA_VERSION, "command": command, **payload}
    return json.dumps(document, default=json_default, ensure_ascii=False)


def emit_json(command: str, payload: dict[str, Any]) -> None:
    """Print exactly one JSON document to stdout."""
    sys.stdout.write(json_document(command, payload))
    sys.stdout.write("\n")
    sys.stdout.flush()


def key_value_table(title: str, rows: dict[str, object]) -> Table:
    """Build a two-column table for a set of labelled values.

    Every cell AND the title go through
    :func:`~llm_eval_lab.reporting.formatters.markup_safe`. Rich parses square
    brackets as style tags, so an unescaped value is not merely mis-styled, it
    is silently DELETED: the redaction marker ``[redacted]`` rendered as nothing
    at all, which made a removed password look like an empty one. Titles
    interpolate a suite name and a price-table id, both authored in untrusted
    data files, so they are exposed in exactly the same way.
    """
    table = Table(title=markup_safe(title), show_header=False, box=None, pad_edge=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    for key, value in rows.items():
        table.add_row(markup_safe(key), "-" if value is None else markup_safe(value))
    return table
