"""Merging suite defaults into cases, selecting cases, and snapshotting.

:class:`~llm_eval_lab.models.BenchmarkSuite` is the authored form and
:class:`~llm_eval_lab.models.ResolvedSuite` is the executable one. Everything
downstream reads only the resolved form, so the merge rules live here, once.

Merge rules, stated so nobody has to infer them from the code:

* ``system`` - the case wins when it sets one, otherwise the suite default.
* ``params`` - merged field by field via
  :meth:`~llm_eval_lab.models.GenerationParams.merged_with`, so a case that
  never mentions ``temperature`` inherits it rather than resetting it.
* ``evaluators`` - a case that lists any evaluators REPLACES the defaults
  rather than appending to them. Appending would make it impossible to opt one
  case out of a default evaluator, and silently scoring a case against a
  criterion its author removed is the worse failure.
* ``tags`` - suite defaults first, then the case's own, de-duplicated with
  order preserved.
* ``metadata`` - shallow merge, case keys winning.
"""

import random
from collections.abc import Sequence

from llm_eval_lab.datasets.hashing import compute_case_hash, compute_suite_hash
from llm_eval_lab.models import (
    BenchmarkCase,
    BenchmarkSnapshot,
    BenchmarkSuite,
    CaseSelection,
    ChatMessage,
    JSONValue,
    ResolvedCase,
    ResolvedSuite,
)
from llm_eval_lab.utils.time import utc_now


def _merge_tags(defaults: Sequence[str], case: Sequence[str]) -> tuple[str, ...]:
    """Concatenate default and case tags, dropping repeats but keeping order."""
    seen: dict[str, None] = {}
    for tag in (*defaults, *case):
        seen.setdefault(tag, None)
    return tuple(seen)


def _messages_for(case: BenchmarkCase) -> tuple[ChatMessage, ...]:
    """Return the case's messages, expanding the single-turn `input` shorthand.

    ``BenchmarkCase`` already guarantees exactly one of the two is set, so
    there is no third branch to handle here.
    """
    if case.messages is not None:
        return case.messages
    return (ChatMessage(role="user", content=case.input or ""),)


def resolve_case(suite: BenchmarkSuite, case: BenchmarkCase) -> ResolvedCase:
    """Merge suite defaults into one case and compute its content hash."""
    messages = _messages_for(case)
    system = case.system if case.system is not None else suite.defaults.system
    params = suite.defaults.params.merged_with(case.params)
    evaluators = case.evaluators or suite.defaults.evaluators
    return ResolvedCase(
        id=case.id,
        case_hash=compute_case_hash(
            case_id=case.id,
            messages=messages,
            system=system,
            expected=case.expected,
            params=params,
            evaluators=evaluators,
        ),
        messages=messages,
        system=system,
        expected=case.expected,
        category=case.category,
        tags=_merge_tags(suite.defaults.tags, case.tags),
        weight=case.weight,
        metadata={**suite.defaults.metadata, **case.metadata},
        params=params,
        evaluators=evaluators,
    )


def resolve_suite(suite: BenchmarkSuite) -> ResolvedSuite:
    """Return the executable, content-addressed form of an authored suite."""
    cases = tuple(resolve_case(suite, case) for case in suite.cases)
    return ResolvedSuite(
        name=suite.name,
        version=suite.version,
        suite_hash=compute_suite_hash(
            schema_version=suite.schema_version,
            name=suite.name,
            version=suite.version,
            cases=cases,
        ),
        cases=cases,
        metadata=suite.metadata,
    )


def select_cases(  # noqa: PLR0913 - one parameter per documented selection filter
    suite: ResolvedSuite,
    *,
    tags: Sequence[str] = (),
    case_ids: Sequence[str] = (),
    limit: int | None = None,
    sample: int | None = None,
    sample_seed: int | None = None,
) -> tuple[tuple[ResolvedCase, ...], CaseSelection]:
    """Apply the case filters in a fixed order and record exactly what was chosen.

    Order is explicit ids, then tags, then sampling, then limit. Sampling uses a
    local :class:`random.Random` seeded from `sample_seed`, so it never touches
    the global RNG and is reproducible when a seed is given.
    """
    selected = list(suite.cases)
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.id in wanted]
    if tags:
        required = set(tags)
        selected = [case for case in selected if required.intersection(case.tags)]
    if sample is not None and sample < len(selected):
        rng = random.Random(sample_seed)  # noqa: S311 - case sampling, not cryptography
        selected = sorted(rng.sample(selected, sample), key=lambda case: case.id)
    if limit is not None:
        selected = selected[:limit]

    chosen = tuple(selected)
    return chosen, CaseSelection(
        tags=tuple(tags),
        case_ids=tuple(case_ids),
        limit=limit,
        sample=sample,
        sample_seed=sample_seed,
        n_selected=len(chosen),
    )


def snapshot_of(suite: ResolvedSuite, *, source_path: str | None) -> BenchmarkSnapshot:
    """Build the persistable snapshot of a resolved suite.

    The whole resolved suite is stored, not a reference to the file, so a run
    stays interpretable after its source changes or is deleted.
    """
    body: JSONValue = suite.model_dump(mode="json")
    return BenchmarkSnapshot(
        suite_hash=suite.suite_hash,
        name=suite.name,
        version=suite.version,
        n_cases=len(suite.cases),
        created_at=utc_now(),
        source_path=source_path,
        body={"suite": body},
    )
