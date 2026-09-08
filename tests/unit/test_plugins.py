"""Opt-in evaluator plugins: four gates, and the API path that must not exist."""

import ast
import builtins
import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from llm_eval_lab.evaluators import plugins
from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.registry import EvaluatorRegistry
from llm_eval_lab.models import (
    EvaluationContext,
    EvaluationResult,
    EvaluatorConfigError,
    ModelResponse,
    ResolvedCase,
)

PROBE_MODULE = "llm_eval_lab_plugin_probe"
PROBE_NAME = "probe_evals"
PROBE_DISTRIBUTION = "acme-evals"
PROBE_VERSION = "1.2.3"


class ProbeParams(BaseModel):
    """Parameters for the probe evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProbeEvaluator(BaseEvaluator[ProbeParams]):
    """A third-party evaluator that always passes."""

    type: ClassVar[str] = "probe_always_passes"
    # `builtins.type` because the `type` ClassVar above shadows the builtin here.
    params_model: ClassVar[builtins.type[BaseModel]] = ProbeParams

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Pass unconditionally."""
        del case, response, context
        return self.verdict(passed=True)


class NotAnEvaluator:
    """Deliberately not a `BaseEvaluator`."""


class FakeDistribution:
    """The distribution metadata `importlib.metadata` would attach to an entry point."""

    name = PROBE_DISTRIBUTION
    version = PROBE_VERSION


@pytest.fixture
def probe_module() -> Any:
    """Install a throwaway module the entry point can genuinely import."""
    module = ModuleType(PROBE_MODULE)
    module.ProbeEvaluator = ProbeEvaluator  # type: ignore[attr-defined]
    module.NotAnEvaluator = NotAnEvaluator  # type: ignore[attr-defined]
    module.PAIR = (ProbeEvaluator,)  # type: ignore[attr-defined]
    sys.modules[PROBE_MODULE] = module
    yield module
    del sys.modules[PROBE_MODULE]


def _entry_point(attribute: str = "ProbeEvaluator", *, with_dist: bool = True) -> EntryPoint:
    """Build a real `EntryPoint` whose `load()` imports the probe module."""
    point = EntryPoint(
        name=PROBE_NAME,
        value=f"{PROBE_MODULE}:{attribute}",
        group=plugins.ENTRY_POINT_GROUP,
    )
    if with_dist:
        vars(point).update(dist=FakeDistribution())
    return point


@pytest.fixture
def advertised(monkeypatch: pytest.MonkeyPatch, probe_module: Any) -> None:
    """Make the probe entry point visible to `importlib.metadata`."""
    del probe_module
    monkeypatch.setattr(plugins, "entry_points", lambda **_: [_entry_point()])


# ---------------------------------------------------------------------------
# the four gates
# ---------------------------------------------------------------------------


def test_a_registered_entry_point_is_not_loaded_when_plugins_are_disabled(
    advertised: None,
) -> None:
    del advertised
    registry = EvaluatorRegistry()
    records = plugins.load_plugins(registry, enabled=False, allowed={PROBE_NAME})
    assert records == ()
    assert registry.types() == ()


def test_an_unlisted_entry_point_is_not_loaded_even_when_plugins_are_enabled(
    advertised: None,
) -> None:
    del advertised
    registry = EvaluatorRegistry()
    records = plugins.load_plugins(registry, enabled=True, allowed={"some_other_name"})
    assert records == ()
    assert registry.types() == ()


def test_an_empty_allowlist_loads_nothing(advertised: None) -> None:
    del advertised
    registry = EvaluatorRegistry()
    assert plugins.load_plugins(registry, enabled=True, allowed=frozenset()) == ()
    assert registry.types() == ()


def test_an_allowlisted_entry_point_loads_and_records_its_provenance(
    advertised: None,
) -> None:
    del advertised
    registry = EvaluatorRegistry()
    records = plugins.load_plugins(registry, enabled=True, allowed={PROBE_NAME})

    assert len(records) == 1
    record = records[0]
    assert record.entry_point == PROBE_NAME
    assert record.distribution == PROBE_DISTRIBUTION
    assert record.version == PROBE_VERSION
    assert record.evaluator_types == ("probe_always_passes",)
    assert registry.types() == ("probe_always_passes",)


async def test_a_loaded_plugin_is_usable_through_the_registry(advertised: None) -> None:
    del advertised
    from llm_eval_lab.models import EvaluatorSpec  # noqa: PLC0415

    registry = EvaluatorRegistry()
    plugins.load_plugins(registry, enabled=True, allowed={PROBE_NAME})
    evaluator = registry.create(EvaluatorSpec(type="probe_always_passes"))
    assert evaluator.type == "probe_always_passes"


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_a_plugin_that_is_not_an_evaluator_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
    probe_module: Any,
) -> None:
    del probe_module
    monkeypatch.setattr(plugins, "entry_points", lambda **_: [_entry_point("NotAnEvaluator")])
    with pytest.raises(EvaluatorConfigError, match=PROBE_NAME):
        plugins.load_plugins(EvaluatorRegistry(), enabled=True, allowed={PROBE_NAME})


def test_a_plugin_that_cannot_be_imported_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = EntryPoint(
        name=PROBE_NAME,
        value="llm_eval_lab_plugin_that_does_not_exist:Thing",
        group=plugins.ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(plugins, "entry_points", lambda **_: [point])
    with pytest.raises(EvaluatorConfigError, match="could not be imported"):
        plugins.load_plugins(EvaluatorRegistry(), enabled=True, allowed={PROBE_NAME})


def test_a_plugin_may_ship_several_evaluators(
    monkeypatch: pytest.MonkeyPatch,
    probe_module: Any,
) -> None:
    del probe_module
    monkeypatch.setattr(plugins, "entry_points", lambda **_: [_entry_point("PAIR")])
    records = plugins.load_plugins(EvaluatorRegistry(), enabled=True, allowed={PROBE_NAME})
    assert records[0].evaluator_types == ("probe_always_passes",)


def test_a_plugin_cannot_shadow_a_built_in_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    probe_module: Any,
) -> None:
    """Silent replacement would change what an existing suite means."""

    class ShadowEvaluator(ProbeEvaluator):
        type: ClassVar[str] = "exact_match"

    vars(probe_module)["Shadow"] = ShadowEvaluator
    monkeypatch.setattr(plugins, "entry_points", lambda **_: [_entry_point("Shadow")])

    from llm_eval_lab.evaluators import BUILTIN_EVALUATORS  # noqa: PLC0415

    registry = EvaluatorRegistry()
    registry.register_all(BUILTIN_EVALUATORS)
    with pytest.raises(EvaluatorConfigError, match="already registered"):
        plugins.load_plugins(registry, enabled=True, allowed={PROBE_NAME})


def test_an_allowlisted_name_nothing_advertises_is_reported(advertised: None) -> None:
    del advertised
    registry = EvaluatorRegistry()
    records = plugins.load_plugins(registry, enabled=True, allowed={PROBE_NAME, "never_installed"})
    assert plugins.unloaded_names({PROBE_NAME, "never_installed"}, records) == ("never_installed",)


# ---------------------------------------------------------------------------
# the API must not be able to reach plugin loading
# ---------------------------------------------------------------------------

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "llm_eval_lab"


def _referenced_names(path: Path) -> set[str]:
    """Return every attribute and imported name a module mentions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("package", ["api", "services"])
def test_no_module_in_the_request_path_mentions_plugin_loading(package: str) -> None:
    """Stronger than patching a call: there is no call site to patch."""
    offenders = [
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in sorted((_SOURCE_ROOT / package).rglob("*.py"))
        if {"load_plugins", "llm_eval_lab.evaluators.plugins"} & _referenced_names(path)
    ]
    assert offenders == []


def test_importing_the_api_does_not_even_import_the_plugin_module() -> None:
    """A subprocess, so no earlier test's import can make this pass by accident."""
    probe = (
        "import importlib, sys;"
        "importlib.import_module('llm_eval_lab.api.app');"
        "print('llm_eval_lab.evaluators.plugins' in sys.modules)"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


async def test_every_api_route_runs_with_plugin_loading_wired_to_explode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The behavioural half: exercise every route with `load_plugins` set to raise."""
    import httpx  # noqa: PLC0415

    from llm_eval_lab.api.app import create_app  # noqa: PLC0415
    from llm_eval_lab.settings import Settings  # noqa: PLC0415

    sentinel = "load_plugins must never be reachable from an API request"

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(sentinel)

    monkeypatch.setattr(plugins, "load_plugins", _explode)

    app = create_app(Settings(_env_file=None))
    paths = sorted({getattr(route, "path", "") for route in app.routes})
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
        for path in paths:
            if not path.startswith("/api") or "{" in path:
                continue
            response = await client.get(path)
            assert sentinel not in response.text
