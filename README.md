# llm-eval-lab

A local-first platform for evaluating and regression-testing LLM behavior: define
reusable benchmark suites, run them against any supported model provider, score
the results with deterministic checks, similarity metrics or an LLM-as-judge,
store every run, and gate a pull request on whether the candidate model regressed
against a baseline.

It is not a thin wrapper around a chat completion call. The parts that make it an
evaluation platform rather than a script are: content-addressed benchmark
suites and price tables, a statistically defensible comparison engine (Wilson
intervals, McNemar's test, Newcombe's Method 10, minimum-sample-size gates), a
documented LLM-as-judge methodology with its own bias disclosures, persistent
run history in SQLite, a REST API and CLI with a stable exit-code contract, and
a dashboard for browsing runs and diffing them visually.

## Why LLM evaluation matters

A model that answers a spot-check prompt correctly today can silently regress
tomorrow: a provider ships a new model snapshot, a prompt template changes, a
system prompt is edited, or a config knob moves. Without a repeatable benchmark
and a statistically honest comparison, "it feels a bit worse" is the only signal
available. `llm-eval-lab` exists to replace that feeling with a number, an
interval around that number, and a deterministic pass/fail gate a CI pipeline
can act on — while being explicit about what the number cannot tell you. See
[`docs/regression-testing.md`](docs/regression-testing.md) for exactly how
honest that number is at small sample sizes.

## Architecture overview

`llm-eval-lab` is a modular monolith, not a set of microservices: one Python
package, one FastAPI app, one CLI, one SQLite (or PostgreSQL-compatible)
database, and a React dashboard that talks to the API. Providers are behind an
abstraction the evaluation engine never bypasses; evaluators are behind a
registry; storage is behind repository protocols. The full layer table, the
frozen contract layer, the async model, the single-worker run manager and its
seam for a future queue, and the project's non-goals all live in
[`docs/architecture.md`](docs/architecture.md) — read that before making a
structural change.

**Reproducibility, stated plainly up front:** "reproducible" here means the
*configuration* that produced a run is fully recoverable — the exact resolved
suite (content-hashed), the exact provider and model, the exact generation
parameters, the exact price table version, the exact library and interpreter
versions. It does **not** mean the model's output will be identical on a
second run. Even at `temperature=0`, hosted providers are not guaranteed to be
deterministic. The fake provider shipped for development and testing *is*
fully deterministic (same input digest, same output, every time), which is
what makes the test suite and the examples below reliable without spending
money on a real model.

## Installation

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

This installs the core package plus every optional extra (`openai`,
`anthropic`, `google` for the vendor SDKs, `embeddings` for the
`semantic_similarity` evaluator's local backend) and the dev toolchain
(pytest, ruff, mypy, import-linter). Ollama is reached over its plain local
HTTP API and needs no extra SDK. To install only what you need in
production:

```bash
uv sync                              # core only: CLI, API, fake provider
uv sync --extra openai --extra anthropic   # plus specific provider SDKs
uv sync --extra all                  # every provider SDK and embeddings backend
```

Copy the example environment file and fill in the credentials for the
providers you actually use — every provider integration is optional and
lazily imported, so an unset key only matters if you try to use that
provider:

```bash
cp .env.example .env
```

Initialize the database (SQLite by default, at
`~/.local/share/llm-eval-lab/llm-eval-lab.db`):

```bash
uv run llm-eval db upgrade
```

## Your first benchmark run, against the fake provider

`llm-eval-lab` ships a deterministic fake provider so you can exercise the
whole pipeline — suite loading, evaluation, statistics, storage, regression
gating — without an API key or a network call. `examples/benchmarks/smoke.yaml`
is a six-case suite exercising each of the five deterministic evaluators once:

```bash
uv run llm-eval run examples/benchmarks/smoke.yaml \
  --provider fake --model fake-1 --fake-mode expected --label "first run"
```

`--fake-mode expected` makes the fake provider return each case's own expected
answer, so this run passes every case — a guaranteed all-green baseline useful
for confirming the pipeline works before you point it at a real model. See
[`docs/benchmark-format.md`](docs/benchmark-format.md) for how to author your
own suite, and [`examples/benchmarks/`](examples/benchmarks/) for six worked
examples covering exact match, JSON extraction, numeric tolerance,
LLM-as-judge grading, similarity scoring and a case that requires multiple
evaluators to agree.

Inspect what happened:

```bash
uv run llm-eval runs                        # list stored runs
uv run llm-eval show <run-id> --cases        # per-case detail
uv run llm-eval metrics <run-id>             # the aggregate rollup
```

Validate a suite file without running it (useful in CI, before spending
anything):

```bash
uv run llm-eval validate examples/benchmarks/smoke.yaml
```

## Running a regression comparison

Run a baseline and a candidate, then gate on the difference:

```bash
uv run llm-eval run examples/benchmarks/smoke.yaml --provider fake --model fake-1 \
  --fake-mode expected --label baseline
uv run llm-eval run examples/benchmarks/smoke.yaml --provider fake --model fake-1 \
  --fake-mode mutate --fake-seed 1 --label candidate

uv run llm-eval compare <baseline-run-id> <candidate-run-id> \
  --thresholds examples/thresholds/ci-thresholds.yaml --json > report.json
echo "exit code: $?"
```

`compare` is the command a CI pipeline keys off: it prints nothing but the
regression report to stdout under `--json` and communicates its verdict
through its exit code. The gate compares point estimates only and is fully
deterministic; the confidence intervals and McNemar p-value in the report are
advisory context for a human, never part of the pass/fail logic. Full details,
including exactly how much a small benchmark can and cannot detect, are in
[`docs/regression-testing.md`](docs/regression-testing.md).

`compare --json` and `report --format json` are deliberately the ONE exception
to the envelope every other command's `--json` emits (see below): both write
the bare `RegressionReport` document itself, with no `schema_version`/`command`
wrapper around it, and nothing else to stdout. This is a contract, not an
oversight — `GET`/`POST /api/compare` returns exactly the same document,
produced by the same serializer, so a dashboard and a CI pipeline reading the
same comparison are never looking at two spellings of it. `compare ... --json
> report.json` and `report ... --format json --out report.json` are meant to
be piped or written straight through, without a caller stripping an envelope
off first. A script that generically checks `.ok` across every command's JSON
output should treat these two as the one deliberate exception.

## CLI usage

Every subcommand accepts `--json` to emit a machine-readable document on
stdout (human-facing output goes to stderr, so `--json` output is always safe
to pipe). Every one of them wraps its payload in the same
`{"schema_version", "command", "ok", ...}` envelope, with one deliberate
exception: `compare` and `report --format json` write the bare regression
report itself, with no envelope, so they match the exact bytes
`GET`/`POST /api/compare` returns — see the note above. Global options
(`--config`, `--db-url`, `--log-level`, `--log-format`, `--no-color`) precede
the subcommand.

| Command | Purpose |
|---|---|
| `llm-eval validate SUITE` | Validate a benchmark file; reports every problem, not just the first. |
| `llm-eval run SUITE --provider P --model M` | Run a suite against a model. Supports `--dry-run`, `--max-cost`, tag/id/sample selection, concurrency and timeout overrides. |
| `llm-eval runs` | List stored runs, newest first, with filters. |
| `llm-eval show RUN_ID` | Show one run's status, totals and, with `--cases`, its per-case results. |
| `llm-eval metrics RUN_ID` | Report the aggregate metrics rollup for one run. |
| `llm-eval compare BASELINE CANDIDATE` | Gate a candidate run against a baseline; drives its exit code from the verdict. |
| `llm-eval report RUN_ID [--compare-to BASELINE]` | Render a Markdown or JSON report for a run or a comparison. |
| `llm-eval export RUN_ID` | Export one run's case results as JSON or CSV. |
| `llm-eval models` | List every model a provider serves or the price table quotes. |
| `llm-eval providers` | List every registered provider and its credential status. |
| `llm-eval evaluators` | List every registered evaluator type, optionally with its parameter schema. |
| `llm-eval pricing show` / `pricing validate` | Inspect or validate the price table. |
| `llm-eval db upgrade` / `db revision` | Apply or create Alembic migrations. |
| `llm-eval config show` | Show the effective configuration and where each value came from. |
| `llm-eval serve` | Serve the REST API and the dashboard. |

### Exit codes

One table, defined once, in `src/llm_eval_lab/cli/main.py`, because a CI
pipeline keys off these and a second definition elsewhere would eventually
disagree with it:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Internal error (an unexpected exception — treat as a bug) |
| `2` | Usage error (bad arguments, or `compare`/`report` given an unknown run id) |
| `3` | Benchmark or threshold-policy validation failed |
| `4` | Regression detected (`compare` verdict `fail`) |
| `5` | Run incomplete (`run` finished in status `partial` or `failed`, or was cancelled by `--max-cost`) |
| `6` | The two runs being compared are not comparable |
| `7` | Warning-severity threshold breaches only, with `compare --fail-on-warning` |
| `130` | Interrupted (`SIGINT` was actually delivered to this process) |

`llm-eval run` sends a graceful cancellation on the first `Ctrl-C` — in-flight
cases finish and are persisted — and a second `Ctrl-C` lets Python's default
handler force the exit.

A run's status can be `cancelled` for two different reasons, and only one of
them is exit `130`. A `--max-cost` budget stopping the run and a `Ctrl-C`
stopping the run both persist whatever cases already completed and both leave
`run.status` as `cancelled` — but no signal was ever delivered to the process
in the budget case, so it exits `5`, the same "didn't get to finish, not the
run's fault" code `partial` and `failed` use. `run.error` says which one
happened: `budget_exceeded` for the budget case, `cancelled` for `Ctrl-C`.
`130` is reserved for the one case that is actually true of it — this process
received `SIGINT`.

## The dashboard

`llm-eval serve` starts the REST API (FastAPI, mounted under `/api`) and
serves the built React dashboard from the same process, at
`http://127.0.0.1:8000` by default:

```bash
uv run llm-eval serve
```

The dashboard lists stored runs, shows a run's per-case results and aggregate
metrics, and renders a baseline-versus-candidate comparison with the same
threshold checks and advisory intervals `llm-eval compare` reports. It is a
read-only view onto the same API a script or CI pipeline would call directly —
`GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/metrics`,
`GET|POST /api/compare`, `GET /api/health` and friends. The API binds to
loopback only unless `LLM_EVAL_API_TOKEN` is set (decision: an unauthenticated
API that can spend money must never be reachable off the local machine by
default).

**Token-protected mode is API-only.** When `LLM_EVAL_API_TOKEN` is set, every
`/api/*` route except `/api/health` requires `Authorization: Bearer <token>`.
The bundled dashboard deliberately never enters, stores or sends that token —
it does not prompt for one, and it does not read one from local storage,
a cookie or the URL. A bearer token here authorises spending money, and the
dashboard is a loopback development tool; putting that secret into a browser
on a network-exposed host would create a new way for it to leak that the
project does not want to open. The dashboard's static files are still served
unauthenticated at `/`, so with a token configured the page loads but every
data request on it returns 401 — the dashboard shows this in a banner
explaining that token mode is API-only rather than a generic error.

There is no way to make the dashboard show data from a token-protected
server. To read such a server, call the API directly with the token:

```bash
curl -H "Authorization: Bearer $LLM_EVAL_API_TOKEN" http://your-host:8000/api/runs
```

To use the dashboard on a remote machine, do not enable token mode at all:
leave the server bound to loopback and reach it over an SSH tunnel, so the
browser talks to a local port and no token is involved anywhere:

```bash
ssh -L 8000:127.0.0.1:8000 your-host
# then open http://127.0.0.1:8000 locally
```

For frontend development against a live API with hot reload:

```bash
uv run llm-eval serve            # terminal 1: API on :8000
cd frontend && npm run dev       # terminal 2: Vite dev server on :5173
```

## Adding a provider

Providers live in `src/llm_eval_lab/providers/` behind a small `BaseProvider`
interface and are registered by name in `providers/registry.py`. The
evaluation engine only ever depends on that interface — `runner`, `evaluators`
and `services` never import a concrete vendor SDK. To add a provider:

1. Implement `BaseProvider` for the new vendor (see `providers/fake.py` for
   the reference shape, and any of `openai_provider.py`,
   `anthropic_provider.py`, `google_provider.py`, `ollama_provider.py` for a
   real integration) — translate `GenerationParams` into the vendor's request
   shape, normalize the response into `ModelResponse`, and map vendor errors
   onto `ProviderError` subtypes so the runner's retry logic classifies them
   correctly.
2. Register it in `ProviderRegistry` with a `ProviderInfo` naming its optional
   install extra and its default credential environment variable — the
   variable *name* only; a `ProviderConfig` never carries a credential value,
   which is what makes it safe to persist, log and return over the API
   verbatim.
3. Gate the vendor SDK import behind the optional extra in `pyproject.toml`,
   so the core install has no hard dependency on it.

See [`docs/providers.md`](docs/providers.md) for the currently supported
providers, their credential variable names, and known normalization gaps.

## Adding an evaluator

Evaluators live in `src/llm_eval_lab/evaluators/` behind `BaseEvaluator` and
are registered by `type` string in `evaluators/registry.py`; a
`BenchmarkCase` references one by that string plus a `params` mapping
validated against the evaluator's own Pydantic params model. To add one:

1. Implement `BaseEvaluator[YourParams]`, returning an `EvaluationResult` that
   distinguishes a **failed** evaluation (the model's answer was wrong) from
   an **errored** one (the evaluator itself could not reach a conclusion —
   the two are never conflated, because collapsing them makes a broken
   evaluator look like a failing model).
2. Register the type in `evaluators/registry.py`.
3. For a genuinely third-party, out-of-tree evaluator, use the entry-point
   plugin mechanism instead of forking the package: plugins are off by
   default, must be explicitly enabled and allowlisted by exact name, are
   never reachable through the API, and — because a benchmark file is
   `yaml.safe_load`ed and can never carry code — a plugin is something an
   operator installs and turns on locally, not something a benchmark author
   can smuggle in.

The full evaluator catalog, including the LLM-as-judge methodology and its
documented limitations, is in [`docs/evaluators.md`](docs/evaluators.md).

## Documentation index

| Document | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layer table, frozen contracts, storage boundary, async decision, run manager, PostgreSQL position, non-goals |
| [`docs/benchmark-format.md`](docs/benchmark-format.md) | Every field of a benchmark suite, defaults merging, multi-evaluator pass rules, hashing |
| [`docs/evaluators.md`](docs/evaluators.md) | All evaluator types, parameters, worked examples, judge limitations |
| [`docs/providers.md`](docs/providers.md) | Supported providers, credential variables, normalization notes |
| [`docs/regression-testing.md`](docs/regression-testing.md) | Threshold file format, gate semantics, CI integration, statistical honesty at small n |

## Development

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run lint-imports
uv run pytest -q
```

## License

MIT. See `LICENSE`.
