"""Rendering the reporting layer's models into the shapes a surface needs.

Three targets, one set of numbers. ``json_out`` produces the documents the CLI
envelope and the HTTP API carry, ``tables`` produces the Rich renderables the
terminal shows, and ``csv_out`` produces rows for a spreadsheet. None of them
computes a statistic: a formatter that derived a figure would be a second
implementation of that figure, and the two would eventually disagree.

The layered contract lets ``cli`` and ``api`` import this package directly,
which is why the renderers live here rather than in either surface.
"""

from llm_eval_lab.reporting.formatters.csv_out import (
    CASE_COLUMNS,
    neutralise_formula,
    write_case_csv,
)
from llm_eval_lab.reporting.formatters.json_out import (
    case_row,
    case_rows,
    metrics_document,
    price_table_document,
)
from llm_eval_lab.reporting.formatters.markdown import (
    advisory_table,
    check_table,
    comparison_markdown,
    markdown_cell,
    metrics_markdown,
)
from llm_eval_lab.reporting.formatters.tables import (
    category_table,
    evaluator_table,
    latency_table,
    markup_safe,
    metrics_table,
    models_table,
    price_entry_table,
)

__all__ = [
    "CASE_COLUMNS",
    "advisory_table",
    "case_row",
    "case_rows",
    "category_table",
    "check_table",
    "comparison_markdown",
    "evaluator_table",
    "latency_table",
    "markdown_cell",
    "markup_safe",
    "metrics_document",
    "metrics_markdown",
    "metrics_table",
    "models_table",
    "neutralise_formula",
    "price_entry_table",
    "price_table_document",
    "write_case_csv",
]
