"""Hatch build hook that bundles the built frontend into the wheel.

Copies ``frontend/dist`` into ``src/llm_eval_lab/api/static/`` before the
wheel is built, if and only if ``frontend/dist`` exists. When no frontend
build is present the hook is a no-op and the wheel builds as a backend-only
package. This lets the packaging metadata stay frozen from phase 0 onward:
the frontend phase adds no packaging change, it only ever produces the
directory this hook already knows to look for.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FrontendStaticBuildHook(BuildHookInterface[Any]):
    """Copy built frontend assets into the package before packaging."""

    PLUGIN_NAME = "frontend-static"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Copy `frontend/dist` into the package's static directory, if present."""
        del version, build_data  # unused: this hook only has file-copy side effects
        root = Path(self.root)
        frontend_dist = root / "frontend" / "dist"
        if not frontend_dist.is_dir():
            return

        static_dir = root / "src" / "llm_eval_lab" / "api" / "static"
        if static_dir.exists():
            shutil.rmtree(static_dir)
        static_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(frontend_dist, static_dir, dirs_exist_ok=True)
