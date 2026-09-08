"""Write the OpenAPI schema to a file without starting a server.

The dashboard generates its TypeScript client from this snapshot, and CI
regenerates it to prove the checked-in copy is not stale. Neither needs a
listening socket, a database or a credential: `create_app` wires the whole
application without touching any of them, and `app.openapi()` reads the routes
off it.

The snapshot is GENERATED, never hand-edited. A field corrected in the JSON but
not in the model would be a lie that survives until someone regenerates.

The script also asserts that every path the API is contracted to expose is
present, so a router accidentally left unregistered fails the export rather than
silently shipping a schema the dashboard will build a missing feature against.
"""
# ruff: noqa: INP001, T201
# `scripts/` is a directory of standalone entry points, not an importable
# package, and printing IS this one's interface: it reports what it wrote, or
# what is missing, to whoever ran it.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_eval_lab.api.app import create_app  # noqa: E402 - after the sys.path bootstrap above
from llm_eval_lab.settings import Settings  # noqa: E402 - after the sys.path bootstrap above

DEFAULT_OUT = REPO_ROOT / "frontend" / "openapi.json"

REQUIRED_PATHS: tuple[str, ...] = (
    "/api/health",
    "/api/version",
    "/api/providers",
    "/api/evaluators",
    "/api/pricing",
    "/api/benchmarks",
    "/api/benchmarks/{name}",
    "/api/benchmarks/validate",
    "/api/runs",
    "/api/runs/{id}",
    "/api/runs/{id}/status",
    "/api/runs/{id}/cases",
    "/api/runs/{id}/cases/{case_id}",
    "/api/runs/{id}/metrics",
    "/api/runs/{id}/cancel",
    "/api/runs/{id}/export",
    "/api/compare",
    "/api/models/summary",
)
"""Every path the API is contracted to expose. Absence here is an export failure."""


def build_schema() -> dict[str, Any]:
    """Return the OpenAPI document for a freshly constructed application.

    Settings are pinned rather than read from the environment. A developer with
    ``LLM_EVAL_API_HOST`` set to a public address would otherwise have the export
    refuse to run, and the schema does not depend on the bind address anyway.
    """
    settings = Settings(_env_file=None, api_host="127.0.0.1")
    schema: dict[str, Any] = create_app(settings).openapi()
    return schema


def missing_paths(schema: dict[str, Any]) -> tuple[str, ...]:
    """Return the contracted paths this schema does not contain."""
    present = set(schema.get("paths", {}))
    return tuple(path for path in REQUIRED_PATHS if path not in present)


def main(argv: list[str] | None = None) -> int:
    """Write the schema to the requested path, or report what is missing from it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write the schema. Defaults to {DEFAULT_OUT.relative_to(REPO_ROOT)}.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the file on disk differs from the schema.",
    )
    args = parser.parse_args(argv)

    schema = build_schema()
    absent = missing_paths(schema)
    if absent:
        print(f"schema is missing {len(absent)} contracted path(s):", file=sys.stderr)
        for path in absent:
            print(f"  {path}", file=sys.stderr)
        return 1

    rendered = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out: Path = args.out
    if args.check:
        current = out.read_text(encoding="utf-8") if out.is_file() else ""
        if current != rendered:
            print(f"{out} is stale; regenerate it with scripts/export_openapi.py", file=sys.stderr)
            return 1
        print(f"{out} is up to date", file=sys.stderr)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {out} with {len(schema.get('paths', {}))} paths", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
