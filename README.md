# llm-eval-lab

A tool that tests how well an AI language model answers a fixed set of
questions, keeps a record of every test, and tells you whether a new model (or
a new prompt) got worse than the one you had before.

It runs on your own machine. It works with OpenAI, Anthropic, Google, Ollama
and any OpenAI-compatible server, and it ships with a fake model so you can try
everything without an API key.

## The idea in plain words

Language models change under you. A provider updates its model, someone edits
a prompt, a setting moves. The model that gave good answers last week may give
worse ones today, and nobody notices until a user complains.

The usual fix is to try a few questions by hand and see if the answers "feel
fine". That does not scale and it does not catch small drops.

`llm-eval-lab` replaces the feeling with a number. You write down the questions
once, with the answers you expect. The tool asks the model every question,
marks each answer right or wrong, and saves the result. Later, you run the same
questions against a new model and the tool compares the two runs. If the new
one scores worse than an amount you set, the tool says so with a pass or fail
that a build pipeline can act on.

## How it works

There are five steps. Each one is a separate piece of the tool, and you can
stop after any of them.

1. **Write a benchmark.** A benchmark is a list of test cases in a YAML file.
   Each case has a question, the answer you expect, and the rule for checking
   the answer.
2. **Run it against a model.** The tool sends every question to the model you
   choose and collects the replies.
3. **Score each reply.** A checker (called an *evaluator*) compares the reply
   with the expected answer and marks it pass or fail.
4. **Store the run.** Every run is saved in a local database with everything
   needed to know what was tested and how.
5. **Compare two runs.** Pick an old run as the baseline and a new run as the
   candidate. The tool checks whether the candidate dropped below your limits
   and returns pass or fail.

The rest of this section explains each step in a little more detail.

### 1. The benchmark file

A case looks like this:

```yaml
- id: capital-exact
  input: What is the capital of France?
  expected: Paris
  evaluators:
    - type: exact_match
```

`input` is what the model is asked. `expected` is the answer you want.
`evaluators` names the rule used to check the reply. You can give a case tags,
a category, and its own settings such as temperature. Shared settings go in a
`defaults` block at the top of the file so you do not repeat them.

The tool computes a fingerprint (a hash) of the whole file. Two runs are only
compared case by case if the cases have not changed, so an edited question is
never silently treated as the same test.

See [`docs/benchmark-format.md`](docs/benchmark-format.md) for every field and
[`examples/benchmarks/`](examples/benchmarks/) for worked examples.

### 2. Running against a model

A *provider* is the code that talks to one vendor's API. The rest of the tool
never talks to a vendor directly, so switching from one model to another is a
change of two command-line flags.

The `fake` provider is built in. It needs no key and no network. It can be told
to return the expected answer for every case (`--fake-mode expected`) so you get
a run that passes everything, or to alter its answers in a repeatable way
(`--fake-mode mutate`) so you get a run with some failures to compare against.

Real providers cost money. You can set a spending cap with `--max-cost`, and
`--dry-run` shows what would be sent without sending it.

### 3. Scoring the replies

There are three kinds of evaluator.

| Kind | What it does | Examples |
|---|---|---|
| Exact rules | Checks the reply with plain string or number logic. Same input, same verdict, every time. | `exact_match`, `contains`, `regex`, `numeric_tolerance`, `json_valid`, `json_schema` |
| Similarity | Gives a score for how close the reply is to the expected text, for cases where wording can vary. | `lexical_similarity`, `semantic_similarity` |
| LLM as judge | Asks a second model to grade the reply against a written rubric, for open-ended answers. | `llm_judge` |

Every evaluator returns a score between 0 and 1, plus pass or fail. A case can
use more than one evaluator. By default the case passes only if every one of
them passes.

An evaluator that *cannot reach a verdict* (for example, the judge model timed
out) is recorded as an error, not as a failure. This matters: without that
distinction a broken checker looks like a bad model.

The judge evaluator has known biases and limits. They are written down in
[`docs/evaluators.md`](docs/evaluators.md).

### 4. Storing the run

Runs go into a SQLite database on your machine (PostgreSQL also works). Each
run records the benchmark fingerprint, the provider and model, the generation
settings, the price table used for cost estimates, the library versions, and
every reply with its scores.

This is what "reproducible" means here. You can always recover exactly *what*
was tested and *how*. It does not mean the model will give the same reply
twice. Hosted models are not guaranteed to be deterministic even at
`temperature=0`. Only the fake provider is.

### 5. Comparing two runs

You give the tool a baseline run, a candidate run, and a thresholds file. The
thresholds file lists rules such as "overall pass rate may not drop by more
than two percentage points" or "error rate must stay below one percent".
`examples/thresholds/ci-thresholds.yaml` is a ready-made one.

The verdict is deterministic. The tool compares the two numbers for each rule
and applies the limit. It also prints confidence intervals and a significance
test next to each result, because a six-case benchmark cannot detect a small
change and the report should say so. Those statistics are advice for a human
reader. They never change the pass or fail.

If a rule has too few cases to mean anything, it reports "insufficient data"
instead of failing the build.

[`docs/regression-testing.md`](docs/regression-testing.md) explains the
thresholds file and how much a small benchmark can and cannot tell you.

## What is in the box

- A Python package with a command-line tool, `llm-eval`.
- A REST API (FastAPI) that exposes the same runs and comparisons.
- A React dashboard for browsing runs and viewing comparisons in a browser.
- One SQLite database by default.

Everything runs as one process. Providers, evaluators and storage each sit
behind a small interface so a new one can be added without touching the rest.
[`docs/architecture.md`](docs/architecture.md) has the full layout; read it
before making a structural change.

## Installation

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

That installs the core package, every provider SDK, the local embeddings
backend for `semantic_similarity`, and the dev tools. For a smaller install:

```bash
uv sync                                    # core only: CLI, API, fake provider
uv sync --extra openai --extra anthropic   # plus specific provider SDKs
uv sync --extra all                        # every provider SDK and embeddings
```

Ollama uses its plain local HTTP API and needs no SDK.

Copy the example environment file and fill in keys for the providers you use.
A missing key only matters if you try to use that provider:

```bash
cp .env.example .env
```

Create the database (SQLite, at
`~/.local/share/llm-eval-lab/llm-eval-lab.db` by default):

```bash
uv run llm-eval db upgrade
```

## Your first run

`examples/benchmarks/smoke.yaml` has six cases, one per exact-rule evaluator.
Run it against the fake provider:

```bash
uv run llm-eval run examples/benchmarks/smoke.yaml \
  --provider fake --model fake-1 --fake-mode expected --label "first run"
```

Every case passes, which confirms the pipeline works before you spend money on
a real model. Look at what happened:

```bash
uv run llm-eval runs                   # list stored runs
uv run llm-eval show <run-id> --cases  # per-case detail
uv run llm-eval metrics <run-id>       # the aggregate numbers
```

Check a benchmark file for mistakes without running it:

```bash
uv run llm-eval validate examples/benchmarks/smoke.yaml
```

## Your first comparison

Make a baseline that passes and a candidate with some failures, then compare:

```bash
uv run llm-eval run examples/benchmarks/smoke.yaml --provider fake --model fake-1 \
  --fake-mode expected --label baseline
uv run llm-eval run examples/benchmarks/smoke.yaml --provider fake --model fake-1 \
  --fake-mode mutate --fake-seed 1 --label candidate

uv run llm-eval compare <baseline-run-id> <candidate-run-id> \
  --thresholds examples/thresholds/ci-thresholds.yaml --json > report.json
echo "exit code: $?"
```

`compare` is the command a build pipeline uses. With `--json` it writes only
the report to stdout and gives its verdict through the exit code (see
[Exit codes](#exit-codes)).

## CLI usage

Every subcommand accepts `--json` for machine-readable output on stdout.
Human-facing text goes to stderr, so `--json` output is always safe to pipe.
Global options (`--config`, `--db-url`, `--log-level`, `--log-format`,
`--no-color`) go before the subcommand.

| Command | Purpose |
|---|---|
| `llm-eval validate SUITE` | Check a benchmark file. Reports every problem, not just the first. |
| `llm-eval run SUITE --provider P --model M` | Run a benchmark against a model. Supports `--dry-run`, `--max-cost`, tag/id/sample selection, concurrency and timeout overrides. |
| `llm-eval runs` | List stored runs, newest first, with filters. |
| `llm-eval show RUN_ID` | Show one run's status and totals. `--cases` adds per-case results. |
| `llm-eval metrics RUN_ID` | Show the aggregate metrics for one run. |
| `llm-eval compare BASELINE CANDIDATE` | Gate a candidate run against a baseline. The exit code carries the verdict. |
| `llm-eval report RUN_ID [--compare-to BASELINE]` | Write a Markdown or JSON report for a run or a comparison. |
| `llm-eval export RUN_ID` | Export one run's case results as JSON or CSV. |
| `llm-eval models` | List the models a provider serves or the price table knows. |
| `llm-eval providers` | List registered providers and whether their credentials are set. |
| `llm-eval evaluators` | List registered evaluator types, optionally with their parameters. |
| `llm-eval pricing show` / `pricing validate` | Inspect or check the price table. |
| `llm-eval db upgrade` / `db revision` | Apply or create database migrations. |
| `llm-eval config show` | Show the effective configuration and where each value came from. |
| `llm-eval serve` | Start the REST API and the dashboard. |

**JSON output shape.** Every command wraps its `--json` payload in the same
`{"schema_version", "command", "ok", ...}` envelope, with one deliberate
exception: `compare --json` and `report --format json` write the bare
regression report with no wrapper. That is the same document `GET`/`POST
/api/compare` returns, so a dashboard and a pipeline reading the same
comparison never see two versions of it. A script that checks `.ok` on every
command's output should treat these two as the exception.

### Exit codes

Defined once, in `src/llm_eval_lab/cli/main.py`, because a pipeline keys off
them:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Internal error (an unexpected exception; treat as a bug) |
| `2` | Usage error (bad arguments, or `compare`/`report` given an unknown run id) |
| `3` | Benchmark or thresholds file failed validation |
| `4` | Regression detected (`compare` verdict `fail`) |
| `5` | Run incomplete (status `partial` or `failed`, or stopped by `--max-cost`) |
| `6` | The two runs being compared are not comparable |
| `7` | Warning-level threshold breaches only, with `compare --fail-on-warning` |
| `130` | Interrupted by `Ctrl-C` |

The first `Ctrl-C` during `llm-eval run` lets in-flight cases finish and be
saved. A second `Ctrl-C` forces the exit.

A run stopped by `--max-cost` and a run stopped by `Ctrl-C` both end with
status `cancelled` and keep the cases that finished. Only the `Ctrl-C` case
exits `130`, because only that one received a signal. The budget case exits
`5`. The run's `error` field says which happened: `budget_exceeded` or
`cancelled`.

## The dashboard

`llm-eval serve` starts the REST API under `/api` and serves the React
dashboard from the same process at `http://127.0.0.1:8000`:

```bash
uv run llm-eval serve
```

The dashboard lists runs, shows per-case results and metrics, and renders a
baseline-versus-candidate comparison with the same checks `llm-eval compare`
reports. It is a read-only view onto the same API a script would call:
`GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/metrics`,
`GET|POST /api/compare`, `GET /api/health` and friends.

The API only listens on the local machine unless `LLM_EVAL_API_TOKEN` is set.
An API that can spend money must never be reachable from the network by
accident.

**Token mode is API-only.** With `LLM_EVAL_API_TOKEN` set, every `/api/*`
route except `/api/health` requires `Authorization: Bearer <token>`. The
dashboard never asks for, stores or sends that token. It is a local
development tool, and putting a secret that authorises spending into a browser
on a network-exposed host would open a new way for it to leak. The dashboard
page still loads, but every data request returns 401, and a banner explains
why.

To read a token-protected server, call the API directly:

```bash
curl -H "Authorization: Bearer $LLM_EVAL_API_TOKEN" http://your-host:8000/api/runs
```

To use the dashboard on a remote machine, leave token mode off and reach the
server over an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 your-host
# then open http://127.0.0.1:8000 locally
```

For frontend development with hot reload:

```bash
uv run llm-eval serve            # terminal 1: API on :8000
cd frontend && npm run dev       # terminal 2: Vite dev server on :5173
```

## Adding a provider

Providers live in `src/llm_eval_lab/providers/` behind the `BaseProvider`
interface and are registered by name in `providers/registry.py`. The runner,
evaluators and services never import a vendor SDK directly.

1. Implement `BaseProvider` for the vendor. `providers/fake.py` shows the
   shape; the OpenAI, Anthropic, Google and Ollama files show real
   integrations. Turn `GenerationParams` into the vendor's request, turn the
   reply into a `ModelResponse`, and map vendor errors onto `ProviderError`
   subtypes so retries are classified correctly.
2. Register it in `ProviderRegistry` with a `ProviderInfo` naming its optional
   install extra and the *name* of its credential environment variable. A
   `ProviderConfig` never holds a credential value, which is what makes it
   safe to store, log and return over the API.
3. Put the vendor SDK behind an optional extra in `pyproject.toml` so the core
   install does not depend on it.

[`docs/providers.md`](docs/providers.md) lists the supported providers, their
credential variables, and known gaps.

## Adding an evaluator

Evaluators live in `src/llm_eval_lab/evaluators/` behind `BaseEvaluator` and
are registered by `type` string in `evaluators/registry.py`. A case names one
by that string and passes `params`, which are validated against the
evaluator's own Pydantic model.

1. Implement `BaseEvaluator[YourParams]` and return an `EvaluationResult`.
   Keep a *failed* result (the answer was wrong) separate from an *errored*
   one (the evaluator could not decide).
2. Register the type in `evaluators/registry.py`.
3. For an evaluator that lives outside this repository, use the entry-point
   plugin mechanism instead of forking. Plugins are off by default, must be
   enabled and allowlisted by exact name, and are never reachable through the
   API. A benchmark file is loaded with `yaml.safe_load` and cannot carry
   code, so a benchmark author cannot smuggle a plugin in.

[`docs/evaluators.md`](docs/evaluators.md) has the full catalog, including the
LLM-as-judge method and its limits.

## Documentation index

| Document | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layers, frozen contracts, storage boundary, async model, run manager, PostgreSQL position, non-goals |
| [`docs/benchmark-format.md`](docs/benchmark-format.md) | Every field of a benchmark file, defaults merging, multi-evaluator pass rules, hashing |
| [`docs/evaluators.md`](docs/evaluators.md) | All evaluator types, parameters, worked examples, judge limitations |
| [`docs/providers.md`](docs/providers.md) | Supported providers, credential variables, normalization notes |
| [`docs/regression-testing.md`](docs/regression-testing.md) | Thresholds file format, gate rules, CI integration, what small samples can show |

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
