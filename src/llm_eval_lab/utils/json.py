"""Canonical JSON serialization.

One spelling of "turn this value into bytes", used by every digest in the
project. Sorted keys, compact separators, UTF-8. Two structurally equal values
therefore always produce the same bytes, which is what makes a content hash a
content hash rather than a hash of an arbitrary rendering.
"""

import json
from typing import Any

_SEPARATORS = (",", ":")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text for a JSON-compatible value.

    Keys are sorted and separators are compact, so the output depends on the
    value's content and never on insertion order or formatting. Non-ASCII text
    is preserved rather than escaped; the bytes are produced by
    :func:`canonical_json_bytes`, which fixes the encoding to UTF-8.

    Raises:
        TypeError: if the value contains something JSON cannot represent. This
            is deliberate. Silently coercing an unexpected object into its
            ``repr`` would make two different values hash the same.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON text for a value, encoded as UTF-8."""
    return canonical_json(value).encode("utf-8")
