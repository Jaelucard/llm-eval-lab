"""Opt-in third-party evaluators, loaded from installed entry points only.

Decision D-CUSTOM, restated because every line of this module follows from it: a
benchmark file is shared data and must never be able to cause code to run. There
is no ``eval``, no ``exec``, no dotted import path, no expression language and no
sandbox. A custom evaluator arrives exactly one way - as an installed
distribution advertising an entry point in the group
``llm_eval_lab.evaluators`` - because installing a package is a trust act the
operator already performed deliberately, and opening a YAML file is not.

Four gates, all of which must be open:

1. ``plugins_enabled`` is false by default, so a fresh install loads nothing.
2. Only names present in ``plugins_allowed`` are loaded. An empty allowlist,
   which is the default, means no plugin loads even with the feature enabled.
   The match is exact: no globs, no prefixes.
3. Loading happens once, at process start: ``cli/main.py`` calls
   :func:`load_default_plugins` from its Typer callback, before any command
   runs. It is never triggered by a request.
4. **Nothing under ``api/`` calls this, and nothing under ``services/`` does
   either** - ``services`` is shared by the CLI and the API, so a call site
   there would be reachable from both. That is not a convention: the HTTP
   surface can be reached by anything that can reach the port, and a plugin is
   arbitrary code from an installed package. ``tests/unit/test_plugins.py``
   asserts the absence of any such call path in either package, and that
   importing the API does not even import this module. The one import-linter
   exemption this module needs (``cli.main -> evaluators.plugins``, in
   ``pyproject.toml``) names ``cli.main`` specifically and nothing wider.

What was loaded is recorded in :class:`~llm_eval_lab.models.PluginRecord` -
entry-point name, distribution, version and the evaluator types it registered -
so a run whose scores depended on third-party code says so in its own
configuration rather than leaving a reader to infer it.
"""

from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.registry import EvaluatorRegistry
from llm_eval_lab.models import EvaluatorConfigError, PluginRecord

ENTRY_POINT_GROUP = "llm_eval_lab.evaluators"
"""The one entry-point group this project reads. Nothing else is consulted."""


def _as_classes(loaded: Any, name: str) -> tuple[type[BaseEvaluator[Any]], ...]:
    """Interpret what an entry point returned as evaluator classes.

    An entry point may point at a single evaluator class or at a sequence of
    them, which is how one distribution ships a family. Anything else is
    refused by name: registering an object that is not an evaluator would fail
    later, during a run, with an error that no longer mentions the plugin.

    Raises:
        EvaluatorConfigError: when the object is not an evaluator class or a
            sequence of them.
    """
    candidates: tuple[Any, ...]
    if isinstance(loaded, type):
        candidates = (loaded,)
    elif isinstance(loaded, Iterable) and not isinstance(loaded, (str, bytes)):
        candidates = tuple(loaded)
    else:
        msg = (
            f"evaluator plugin {name!r} resolved to {type(loaded).__name__}, which is neither "
            f"an evaluator class nor a sequence of them"
        )
        raise EvaluatorConfigError(msg)

    for candidate in candidates:
        if not (isinstance(candidate, type) and issubclass(candidate, BaseEvaluator)):
            msg = (
                f"evaluator plugin {name!r} offered {candidate!r}, which is not a subclass of "
                f"BaseEvaluator"
            )
            raise EvaluatorConfigError(msg)
        if not getattr(candidate, "type", None):
            msg = f"evaluator plugin {name!r} offered a class with no `type` name"
            raise EvaluatorConfigError(msg)
    return candidates


def _load_one(point: EntryPoint) -> Any:
    """Import the object an entry point names.

    Raises:
        EvaluatorConfigError: when the import fails for any reason. A plugin is
            third-party code and can raise anything at import time; the failure
            is translated so it names the plugin, and is re-raised rather than
            swallowed, because a run that silently lost an evaluator would
            report a pass rate over the wrong set of checks.
    """
    try:
        return point.load()
    except Exception as exc:
        msg = f"evaluator plugin {point.name!r} could not be imported: {exc}"
        raise EvaluatorConfigError(msg) from exc


def discover_entry_points() -> tuple[EntryPoint, ...]:
    """Return every entry point advertised in this project's evaluator group."""
    return tuple(entry_points(group=ENTRY_POINT_GROUP))


def load_plugins(
    registry: EvaluatorRegistry,
    *,
    enabled: bool,
    allowed: Iterable[str],
) -> tuple[PluginRecord, ...]:
    """Register allowlisted evaluator plugins into `registry`.

    Configuration is passed in rather than read here, so this function never
    touches the environment and a test exercises it by argument alone.

    Args:
        registry: The registry the loaded evaluator types are added to. The
            registry itself refuses a name that is already taken, so a plugin
            cannot shadow a built-in evaluator and quietly change what an
            existing suite means.
        enabled: Whether plugin loading is permitted at all. False is the
            default everywhere and short-circuits before any entry point is
            even enumerated.
        allowed: Exact entry-point names permitted. Empty means none.

    Returns:
        One record per loaded plugin, in allowlist-independent discovery order,
        for :attr:`~llm_eval_lab.models.RunConfig.plugins`.

    Raises:
        EvaluatorConfigError: when an allowlisted entry point cannot be
            imported, does not resolve to evaluator classes, or collides with an
            already-registered type. All three are configuration faults the
            operator must see, not conditions to work around.
    """
    if not enabled:
        return ()
    permitted = frozenset(allowed)
    if not permitted:
        return ()

    records: list[PluginRecord] = []
    for point in discover_entry_points():
        if point.name not in permitted:
            continue
        classes = _as_classes(_load_one(point), point.name)
        registry.register_all(classes)
        distribution = point.dist
        records.append(
            PluginRecord(
                entry_point=point.name,
                distribution="unknown" if distribution is None else distribution.name,
                version="unknown" if distribution is None else distribution.version,
                evaluator_types=tuple(sorted(item.type for item in classes)),
            )
        )
    return tuple(records)


def unloaded_names(allowed: Iterable[str], loaded: Sequence[PluginRecord]) -> tuple[str, ...]:
    """Return allowlisted names that no installed distribution advertises.

    An allowlist entry with nothing behind it is almost always a typo or a
    package that failed to install, and silently loading four of five requested
    plugins is how a suite ends up scored by fewer checks than its author
    believed.
    """
    return tuple(sorted(frozenset(allowed) - {record.entry_point for record in loaded}))


def load_default_plugins(*, enabled: bool, allowed: Iterable[str]) -> tuple[PluginRecord, ...]:
    """Load allowlisted plugins into the process-wide registry, by name only.

    The ONE function ``cli/main.py`` calls, and the reason this module offers
    it rather than making the caller pass a registry of its own: the
    import-linter contract that keeps ``cli`` off persistence and providers
    also names ``evaluators`` among the modules it may not import DIRECTLY,
    with exactly one exemption carved out for this module
    (``pyproject.toml``, the "api and cli reach persistence and providers
    only through services" contract). Importing
    :func:`~llm_eval_lab.evaluators.default_registry` here, rather than in
    ``cli/main.py``, is what keeps that exemption to the single edge it
    names - ``cli.main -> evaluators.plugins`` - instead of needing a second
    one for ``cli.main -> evaluators``.
    """
    from llm_eval_lab.evaluators import default_registry  # noqa: PLC0415 - see the docstring above

    return load_plugins(default_registry(), enabled=enabled, allowed=allowed)


__all__ = [
    "ENTRY_POINT_GROUP",
    "discover_entry_points",
    "load_default_plugins",
    "load_plugins",
    "unloaded_names",
]
