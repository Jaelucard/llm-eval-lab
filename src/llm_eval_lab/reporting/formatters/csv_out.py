"""CSV rows, for the spreadsheet a human actually opens.

One column order, declared once in :data:`CASE_COLUMNS`, so an export written
today and one written after the next release line up in the same diff. Missing
values are written as an EMPTY FIELD rather than as ``0`` or the word ``None``:
a spreadsheet reads an empty numeric cell as absent and a zero as a
measurement, and this file is where that distinction survives or is lost.

**Text cells are neutralised against formula injection.** A benchmark file is
shared data and is treated as untrusted (decision D-CUSTOM), and a case's
``category`` and ``tags`` carry no pattern restriction at all. A category of
``=HYPERLINK("http://attacker.example","Q3 results")`` becomes a live formula
the moment the export is opened in Excel or LibreOffice, so any TEXT value
beginning with one of :data:`FORMULA_TRIGGERS` is prefixed with a single quote,
which is the standard mitigation. Numbers and booleans are never prefixed: they
are produced by this codebase rather than by a benchmark author, and quoting a
negative cost would turn a real measurement into text.
"""

import csv
from collections.abc import Sequence
from typing import TextIO

from llm_eval_lab.models import CaseResult, JSONValue
from llm_eval_lab.reporting.formatters.json_out import case_row

CASE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "case_id",
    "case_hash",
    "status",
    "passed",
    "score",
    "attempts",
    "category",
    "tags",
    "latency_ms",
    "total_latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "total_cost",
    "judge_cost",
    "priced",
    "unpriced_reason",
    "error_kind",
    "started_at",
    "completed_at",
)
"""The case export's column order. Stable across releases by intent."""


FORMULA_TRIGGERS: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")
"""Leading characters a spreadsheet reads as the start of a formula."""

FORMULA_GUARD = "'"
"""Prefix that forces a spreadsheet to treat the cell as literal text."""


def neutralise_formula(text: str) -> str:
    """Return `text` prefixed so a spreadsheet cannot execute it as a formula."""
    if text.startswith(FORMULA_TRIGGERS):
        return FORMULA_GUARD + text
    return text


def _cell(value: JSONValue) -> str:
    """Render one value for a CSV cell, leaving an unknown one empty.

    Only a string is neutralised. A number reaching here was computed by this
    codebase, never authored in a benchmark file, so guarding it would only
    corrupt a legitimate negative value into text.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return neutralise_formula(value)
    return str(value)


def write_case_csv(results: Sequence[CaseResult], stream: TextIO) -> None:
    """Write case results to `stream` as CSV, header first.

    The line terminator is set to a bare newline because the module default is
    a carriage return plus a newline, which on a Unix terminal shows up as a
    trailing control character in every piped line.
    """
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CASE_COLUMNS)
    for result in results:
        row = case_row(result)
        writer.writerow([_cell(row.get(column)) for column in CASE_COLUMNS])


__all__ = [
    "CASE_COLUMNS",
    "FORMULA_GUARD",
    "FORMULA_TRIGGERS",
    "neutralise_formula",
    "write_case_csv",
]
