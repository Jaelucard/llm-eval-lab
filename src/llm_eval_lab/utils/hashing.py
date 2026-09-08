"""The one content-digest primitive the whole project hashes with.

Lives in ``utils`` rather than in ``datasets`` because three sibling packages
need it - suite hashing, price-table identity and judge-rubric identity - and
the layered dependency contract makes them independent of each other. A shared
primitive at the bottom is the only placement that does not create a sideways
import.
"""

import hashlib

from llm_eval_lab.utils.json import canonical_json_bytes

HASH_PREFIX = "sha256:"
"""Every digest is prefixed, so a bare hex string is never mistaken for one."""


def canonical_digest(payload: object) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON encoding of `payload`."""
    return f"{HASH_PREFIX}{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"
