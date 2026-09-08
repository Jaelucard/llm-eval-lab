# Architecture

`llm-eval-lab` is a modular monolith: one Python package, one FastAPI process,
one CLI, one database, one dashboard build. There are no microservices, no
message broker and no distributed job queue. This document describes the
layering that keeps that monolith clean, the parts of the contract that are
frozen, where the storage and process boundaries sit, and what is
deliberately not built.

## Layer table

Layers are numbered lowest first. A module may import only from strictly
lower layers; this is enforced mechanically, not by convention — an
`import-linter` layered contract runs in CI beside Ruff and mypy, and a Ruff
`flake8-tidy-imports.banned-api` rule is a second net.

| Layer | Package | May import | Must NOT import |
|---|---|---|---|
| 0 | `utils`, `observability` | stdlib, structlog | anything else in this project |
| 1 | `models` | pydantic, stdlib, `utils` | FastAPI, Typer, SQLAlchemy, httpx, any vendor SDK, every other project package |
| 2 | `datasets`, `pricing` | `models`, `utils` | providers, evaluators, runner, storage, services, api, cli |
| 2 | `providers` | `models`, `utils`, `observability`, httpx, vendor SDKs | storage, evaluators, runner, services, api, cli, datasets |
| 3 | `evaluators` | `models`, `datasets`, `utils` | concrete providers (only the `Provider` Protocol), storage, api, cli |
| 3 | `reporting` | `models`, `pricing`, `utils` | storage, providers, api, cli |
| 4 | `runner` | `models`, `providers.registry`, `evaluators.registry`, `datasets`, `pricing`, `reporting.aggregate` | storage concretes (Protocols only), api, cli |
| 4 | `storage` | `models`, `utils`, SQLAlchemy, Alembic | runner, providers, evaluators, reporting, api, cli, services |
| 5 | `services` | everything in layers 0–4 | FastAPI, Typer, HTTP or console concerns |
| 6 | `api`, `cli` | `services`, `models`, `reporting.formatters` | SQLAlchemy, provider SDKs, business logic of their own |

Stated plainly, the rules this table exists to enforce:

- `providers` never persists anything and never imports `storage`. A provider
  integration's only job is turning a `ProviderRequest` into a `ModelResponse`
  or a `ProviderError`.
- An API route handler validates a DTO, calls a service, and maps the result
  onto an HTTP response. A route body longer than roughly 15 lines is a review
  defect — business logic belongs in `services`.
- `models` is importable with only `pydantic` installed; a test asserts this
  directly, because it is what lets the domain contract be reused (or read) by
  something outside this process without dragging in FastAPI or SQLAlchemy.
- `runner` and `services` reach persistence exclusively through the
  repository Protocols defined in `models/protocols.py`. No SQLAlchemy ORM
  instance ever crosses a repository boundary into `runner`, `evaluators` or
  the API layer.
- Evaluators receive a `Provider` through `EvaluationContext` rather than
  constructing one themselves. The LLM-as-judge evaluator in particular never
  builds a vendor client directly — it goes through the same provider
  abstraction candidate models do.

## Package layout

```
src/llm_eval_lab/
├── models/          # frozen contract layer — pure domain, pydantic only
├── providers/        # fake + real vendor integrations behind BaseProvider
├── datasets/          # suite loading, defaults resolution, content hashing
├── evaluators/       # deterministic, JSON, similarity and judge evaluators
├── runner/            # RunEngine, retry/backoff, rate limiting, progress
├── pricing/           # price table loading and cost calculation
├── reporting/          # statistics, aggregation, regression engine, formatters
├── storage/            # SQLAlchemy engine, ORM, mappers, repositories
├── services/           # the only orchestration layer: run/comparison/catalog/metrics services
├── api/                # FastAPI app, routers, DTOs, RFC-9457 problem details
├── cli/                # Typer app, exit codes, command modules
├── observability/     # structlog config and the credential-scrubbing processor
└── utils/              # canonical JSON, timing helpers, safe path resolution
```

`frontend/` (the React/Vite dashboard), `tests/{unit,integration,e2e}`,
`examples/{benchmarks,thresholds}` and `docs/` sit alongside `src/` at the
repository root.

## Packaging the dashboard into a wheel

`frontend/dist` (the dashboard's built output) is gitignored, and
`hatch_build.py` is a build hook, not a builder: it copies `frontend/dist`
into `src/llm_eval_lab/api/static/` when the directory exists, and does
nothing otherwise. It never shells out to `npm` itself, which would make a
Python wheel build depend on Node.js and network access being available
wherever `pip install` happens to trigger one.

That copy step needs `frontend/dist` to already exist on disk, so **build the
dashboard before packaging**, from a checkout:

```sh
cd frontend && npm install && npm run build && cd ..
uv build          # or: uv build --wheel
```

Both invocations of `uv build` now bundle the dashboard when `frontend/dist`
is present at build time, but they get there differently, which matters if
you are packaging from something other than a live checkout:

- `uv build --wheel` builds the wheel directly from the checkout, so the hook
  sees `frontend/dist` on disk immediately. This is the simplest path and the
  one to reach for locally.
- `uv build` (no flag) builds a source distribution FIRST and then builds the
  wheel from *that* extracted sdist, in an isolated temporary directory. The
  hook therefore needs `frontend/dist` to travel inside the sdist itself, not
  just exist in the original checkout - which is why the sdist's declared
  `artifacts` include `/frontend/dist/**/*` (alongside an ordinary `include`
  entry, because `hatch_build.py`'s target directory is gitignored and
  hatchling's `include` patterns do not pull in VCS-ignored paths on their
  own). Only the BUILT output travels this way, never `frontend/`'s source:
  the sdist stays buildable by a `pip install` that has no Node.js and no
  network access, exactly as it did before.

Neither path runs without a pre-built `frontend/dist`. A checkout, or an
sdist, with no dashboard build present still produces a completely valid,
backend-only wheel - `/` serves the placeholder page in that installation,
explaining how to build one.

## The frozen contract layer

`src/llm_eval_lab/models/**` and `src/llm_eval_lab/settings.py` are the
project's frozen contract: every domain type (`BenchmarkSuite`,
`ResolvedSuite`, `EvaluationResult`, `Run`, `CaseResult`, `AggregateMetrics`,
`RegressionReport`, `PriceTable`, `CostBreakdown`, the repository Protocols,
and everything else layer 1 exports) lives there, pure Pydantic with no
framework dependency. Every other layer treats these types as given.

That layer is frozen in the sense that changing a shipped field's meaning,
removing a field, or altering a validation rule is a breaking change to every
consumer at once — the runner, the storage mappers, the API DTOs and the CLI
renderers all read the same objects. In practice this means: additive changes
(a new optional field with a safe default) are low-risk; changing what an
existing field means, or its shape, requires updating every layer that reads
it in the same change, not incrementally.

Two properties fall out of this design directly:

- **`passed` vs. `status`.** `EvaluationResult.status` reports whether the
  *evaluator* reached a conclusion (`PASSED`, `FAILED`, `ERROR`, `SKIPPED`);
  `EvaluationResult.passed` reports whether the *case* met the bar. These are
  validated to stay consistent with each other on the model itself (an
  `ERROR` status always carries `passed=None` and a populated `error`; a
  `FAILED` status always carries `passed=False`), so a broken judge can never
  be silently read as a failing model.
- **Money is never `0` for "unknown".** `CostBreakdown.total_cost` is `None`
  when a price is unavailable, never `0`. A `0` reads as "this model is
  free," which is a claim the system has no basis for making; `None` reads as
  "unknown," which is the truth. Every cost aggregate is reported alongside a
  coverage fraction (`cost_coverage`, `token_usage_coverage`) so a mean
  computed over a fraction of cases is never presented with the confidence of
  one computed over all of them.

## The storage boundary

SQLite is the zero-configuration default (`sqlite+aiosqlite:///…`, WAL mode,
a `busy_timeout`, short transactions, a single writer path through the
repository layer). The schema, mappers and query patterns are written to be
PostgreSQL-compatible by design — see below — but SQLite is what every test,
every example in this repository, and the default `llm-eval db upgrade` path
actually exercises.

The storage boundary is the repository Protocols in `models/protocols.py`.
`runner` and `services` depend on those Protocols, never on
`storage`'s SQLAlchemy concretes; an ORM row is mapped onto a domain model
(and back) entirely inside `storage/mappers.py`, and nothing outside
`storage` ever holds a live ORM instance. This is what makes the storage
layer swappable in principle without touching the runner, the evaluators or
the API.

## The async decision

The runner, storage, services and API layers are async throughout
(`asyncio`, SQLAlchemy's async engine, `httpx` async clients). Providers issue
concurrent requests up to a run's configured `concurrency`; a case holds its
concurrency slot across both generation and evaluation. The one CLI-level
synchronous boundary is `run_async()` in `cli/main.py` — the single
`asyncio.run()` entry point for the whole command layer, so a command function
itself stays ordinary synchronous Typer code and only ever awaits inside the
one coroutine it hands to `run_async`.

The one deliberate exception to "everything async" is the regex evaluator's
match execution, which runs in a **separate process**, not a thread, under a
wall-clock timeout. CPython's `re` engine executes a match in a single C call
that never returns to the interpreter loop and never releases the GIL, so a
catastrophic-backtracking pattern running on a thread would stall the entire
event loop — including the timeout that was supposed to stop it — rather than
merely blocking one coroutine. Killing a subprocess is the only mechanism
CPython offers that reliably stops a runaway match in progress. The cost is a
bounded ~15 ms interpreter startup per regex evaluation, which is
negligible against the network latency of the model call the evaluation is
paired with.

## The run manager and its single-worker assumption

`services/run_manager.py` is the in-process seam between the API's
fire-and-forget `POST /api/runs` and the actual execution of a run: it holds
run state in memory while a run is in flight, is what `GET
/api/runs/{id}/status` polls, and is what a `POST /api/runs/{id}/cancel`
signals.

**This is a single-worker, single-process design**, stated as a decision, not
an oversight. There is no distributed job queue, no worker pool and no
multi-process run execution in this system. A run started through the API
executes inside the same process that is serving that API request; two runs
started concurrently share that process's event loop and its configured
concurrency budget. `RunManager` is the named seam where a real queue (Celery,
RQ, an external worker fleet reading from a durable queue) would go if this
ever needed to scale past one machine — the API layer already talks to
`RunManager` through a narrow interface, so replacing the in-process
implementation with one that dispatches to external workers would not require
changing the API routers or the CLI's `run` command. That replacement is not
built; the seam is.

Status is **polled**, not pushed. There is no token-by-token streaming and no
WebSocket live progress in this system (see Non-goals below) — the CLI's own
progress bar and the dashboard's run-status view both work by polling the same
`GET /api/runs/{id}/status` endpoint the run manager backs.

## PostgreSQL: compatible by design, not shipped

`R-ARCH-06` requires SQLite as the zero-configuration default with an
architecture *compatible* with PostgreSQL — not that PostgreSQL is actually
run anywhere in this project. Concretely: the schema avoids SQLite-only
types, migrations are written through SQLAlchemy's dialect-neutral
constructs where practical, and the one place a dialect difference is
unavoidable (SQLite returns naive datetimes, PostgreSQL's `TIMESTAMPTZ`
returns timezone-aware ones) is handled explicitly in the storage mapper
rather than left as an assumption. What this project does **not** do: there is
no PostgreSQL CI job, no PostgreSQL integration test, and "runs against
PostgreSQL" is not a completion claim anywhere in this repository. It is a
design position — "nothing here would need to be re-architected to add
Postgres support" — not a tested one.

## Non-goals

Reproduced here, verbatim in intent, so every omission below reads as a
decision made deliberately rather than a gap nobody noticed.

- Multi-user accounts, authentication beyond an optional local bearer token,
  roles, tenancy.
- Distributed job queue, worker pool, or multi-process run execution.
  `RunManager` is the named seam; the single-worker assumption is documented
  above.
- Cloud deployment, Docker Compose stacks, Kubernetes. A single optional
  Dockerfile is acceptable only if it costs nothing to maintain.
- Actually shipping and operating PostgreSQL. Compatibility by design only —
  see above; no Postgres CI job is a completion gate.
- Token-by-token streaming and WebSocket live progress. Status is polled.
- Agentic, tool-calling or multi-turn conversational evaluation beyond a
  static message list.
- Fine-tuning, synthetic dataset generation, human annotation interfaces.
- Response caching, vector stores, retrieval-augmented evaluation.
- OpenTelemetry export, metrics backends, alerting. Structured logs with a
  documented extension seam are what ships instead.
- Model routing, fallback chains, prompt experiment management.
- Internationalization, theming systems, dashboard customization.
- A public plugin registry or marketplace for evaluator plugins.

**Future work, explicitly excluded from any completion claim** — named so a
reviewer can see each was considered and deferred, not missed, and none of
these is ever described as "done" or "partially done":

1. An AST-allowlist expression evaluator for arbitrary custom scoring logic —
   a constrained expression language with no imports, no attribute access, no
   calls outside a whitelist, and node-count/wall-clock bounds. Not built;
   the entry-point plugin mechanism (opt-in, allowlisted, never
   API-reachable) is what v1 ships instead for custom evaluation logic.
2. Bootstrap confidence intervals for latency percentile deltas and score
   deltas. If built later: fixed seed, capped at 500 resamples, offered only
   when both runs have `n >= 50`, behind an explicit opt-in flag.
3. A local `sentence-transformers` embedding backend for `semantic_similarity`.
   Deliberately not shipped in the base install; adding it later is a new
   backend value, not a redesign.
4. `MetricCheck.require_significant`. The field exists in the frozen
   contract for forward compatibility; setting it raises
   `EvaluatorConfigError` in this version.
5. Pairwise judge mode with position swapping. `JudgeConfig.swap_positions`
   exists in the contract; only single-answer grading is implemented.
6. An HTML report formatter. `llm-eval report` emits Markdown and JSON only;
   `--format html` is not accepted and is not offered in `--help`. Markdown
   already renders everywhere a reviewer reads (a PR, a pager, a chat
   window) and JSON already serves machines — an HTML path would add a third
   rendering surface to secure against the untrusted model output it would
   embed, for presentation polish alone.
7. `DELETE /api/runs/{id}`. No dashboard view uses it and no CLI command
   needs it; a destructive endpoint on an unauthenticated-by-default local
   API is a liability that buys nothing. Runs are removed by deleting the
   database file or by direct SQL.
8. `llm-eval pricing reprice` — recomputing a historical run's cost against a
   newer price table. The unit prices actually applied are frozen onto every
   `CaseResult.cost` at run time, so historical costs stay correct and
   auditable without this feature; it simply is not offered.

## Reproducibility, precisely

Repeated here because it is easy to misread the word "reproducible" in an
evaluation platform: **reproducible means the configuration that produced a
run is fully recoverable, not that the model's output will repeat.**
`RunConfig` is persisted verbatim — the resolved suite's content hash, the
exact provider and model, the exact generation parameters, the exact price
table identity, the library and interpreter versions — so anyone can see
precisely what a run would need in order to be re-executed identically. What
is not guaranteed, and cannot be guaranteed by anything in this system, is
that re-executing it produces byte-identical model output. Even at
`temperature=0`, hosted providers are not contractually deterministic across
calls, infrastructure changes, or time. The fake provider used throughout
development and in this repository's examples *is* fully deterministic by
construction (every stochastic quantity derives from a digest of the
request, seeded into a local `random.Random`), which is precisely what makes
it suitable for tests and worked examples — and precisely why it is not a
substitute for the reproducibility caveat when a real provider is in use.
