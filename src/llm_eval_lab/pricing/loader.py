"""Loading a versioned price table from an external data file.

Prices live in data, not in code, and every table carries an ``id``, a
``version`` and a content hash. A run records all three, so a cost figure stays
interpretable after a vendor changes its pricing page: the numbers a run was
priced with are recoverable even when the shipped table has moved on.

**Prices are written as quoted strings.** ``"0.15"`` parses into
``Decimal("0.15")`` exactly; the YAML float ``0.15`` parses into a binary
double first and carries that error into every cost computed from it. The
loader accepts a bare number and converts it through ``str`` so an unquoted
value is still exact as written, but the shipped table quotes everything.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm_eval_lab.models import PriceTable, PricingError
from llm_eval_lab.utils.hashing import canonical_digest

DEFAULT_PRICE_TABLE_PATH = Path(__file__).parent / "data" / "prices.yaml"
"""The table shipped inside the package. Overridable per run with `--prices`."""

_PRICE_FIELDS = ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok")


def _exact_decimal(value: Any) -> Any:
    """Convert a numeric price to an exactly-as-written :class:`Decimal`."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return value


def _normalize_entry(entry: Any) -> Any:
    """Return one raw price entry with its numeric fields made exact."""
    if not isinstance(entry, dict):
        return entry
    return {
        key: (_exact_decimal(value) if key in _PRICE_FIELDS else value)
        for key, value in entry.items()
    }


def build_price_table(document: dict[str, Any], *, source: str | None = None) -> PriceTable:
    """Validate a parsed price document into a content-addressed table.

    Raises:
        PricingError: when the document does not describe a usable table.
    """
    raw_models = document.get("models", [])
    if not isinstance(raw_models, list):
        msg = f"price table 'models' must be a list, got {type(raw_models).__name__}"
        raise PricingError(msg)

    payload = {
        "id": document.get("id", "unnamed"),
        "version": document.get("version", "0"),
        "currency": document.get("currency", "USD"),
        "source": document.get("source", source),
        "models": [_normalize_entry(entry) for entry in raw_models],
    }
    content_hash = canonical_digest(
        {
            "id": payload["id"],
            "version": payload["version"],
            "currency": payload["currency"],
            "models": [
                {key: str(value) for key, value in entry.items()}
                for entry in payload["models"]
                if isinstance(entry, dict)
            ],
        }
    )
    try:
        return PriceTable.model_validate({**payload, "content_hash": content_hash})
    except ValidationError as exc:
        msg = f"price table is invalid: {exc}"
        raise PricingError(msg) from exc


def load_price_table(path: Path | str | None = None) -> PriceTable:
    """Load a price table from `path`, or the table shipped with the package.

    Raises:
        PricingError: when the file is missing, unparseable, or invalid.
    """
    resolved = Path(path) if path is not None else DEFAULT_PRICE_TABLE_PATH
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read price table {resolved}: {exc.strerror or exc}"
        raise PricingError(msg) from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"price table {resolved} is not valid YAML: {exc}"
        raise PricingError(msg) from exc
    if not isinstance(document, dict):
        msg = f"price table {resolved} must be a mapping, got {type(document).__name__}"
        raise PricingError(msg)
    return build_price_table(document, source=str(resolved))
