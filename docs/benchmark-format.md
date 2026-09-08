# Benchmark format

A benchmark suite is a YAML or JSON file validated against
`llm_eval_lab.models.BenchmarkSuite`. Loading is strict: the file is parsed
with `yaml.safe_load` only (nothing in a benchmark file can ever execute
code), it is size-capped before parsing, and every validation problem in the
file is reported at once rather than stopping at the first one.

```yaml
schema_version: 1
name: my-suite
version: "1"
description: An example suite.
defaults:
  system: You are a terse assistant.
  params:
    temperature: 0.0
  tags: [smoke]
cases:
  - id: case-1
    input: What is the capital of France?
    expected: Paris
    evaluators:
      - type: exact_match
```

## `BenchmarkSuite` fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | `1` | yes (defaults to `1`) | The only accepted value today; present so a future format change can be detected rather than silently misparsed. |
| `name` | string | yes | Pattern `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. |
| `version` | string | no (default `"1"`) | Free-form; not compared for equality anywhere except `compare --require-same-suite` reasoning. |
| `description` | string \| null | no | |
| `defaults` | `SuiteDefaults` | no | Values every case inherits unless it overrides them. See below. |
| `cases` | list of `BenchmarkCase` | yes, at least one | Case `id`s must be unique within the suite; a duplicate is a validation error naming every id that repeats. |
| `metadata` | mapping | no | Free-form JSON-compatible data, carried through to the persisted snapshot but not interpreted by the engine. |

## `SuiteDefaults` fields

| Field | Type | Notes |
|---|---|---|
| `system` | string \| null | Default system prompt. |
| `params` | `GenerationParams` | Default decoding parameters. |
| `evaluators` | list of `EvaluatorSpec` | Default evaluator set. |
| `tags` | list of strings | Default tags. |
| `metadata` | mapping | Default metadata. |

## `BenchmarkCase` fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | Pattern `^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`. Identifies the case across runs for pairing (see Hashing, below). |
| `input` | string \| null | exactly one of `input`/`messages` | Single-turn shorthand: becomes one `user` message. |
| `messages` | list of `ChatMessage` \| null | exactly one of `input`/`messages` | Multi-turn form, for a case that needs a scripted conversation. |
| `system` | string \| null | no | Overrides `defaults.system` for this case. |
| `expected` | any JSON value | no | The reference answer. Several evaluators read this by default; most accept a `params.expected` override instead. |
| `category` | string \| null | no | A single classification bucket, used for `by_category` metrics and category-scoped regression gates. |
| `tags` | list of strings | no | Zero or more free-form labels, used for `by_tag` metrics and `--select-tag` filtering. Merged with, not replacing, `defaults.tags`. |
| `weight` | float ≥ 0 | no (default `1.0`) | Reserved for weighted aggregation; the shipped metrics do not currently weight by it. |
| `metadata` | mapping | no | Free-form, carried through but not interpreted. |
| `params` | `GenerationParams` \| null | no | Per-case override, merged onto `defaults.params` — see Defaults merging, below. |
| `evaluators` | list of `EvaluatorSpec` | no | Per-case evaluator set. **Replaces**, does not merge with, `defaults.evaluators` when present — see below. |

A case must supply **exactly one** of `input` or `messages`. Supplying
neither, or both, is a validation error naming the case.

## `EvaluatorSpec` fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | Must match a registered evaluator type (`llm-eval evaluators` lists them). |
| `id` | string \| null | no | Defaults to `type` when omitted; distinguishes two instances of the same evaluator type on one case (e.g. two `regex` checks with different patterns) in per-evaluator metrics. |
| `weight` | float ≥ 0 | no (default `1.0`) | Reserved for weighted aggregation, same status as case `weight`. |
| `required` | bool | no (default `true`) | See "the multi-evaluator pass rule," below. |
| `on_error` | `"error" \| "fail" \| "skip"` | no (default `"error"`) | How a case-level verdict treats this evaluator erroring: `error` propagates the error status, `fail` treats an evaluator error as a failed case, `skip` excludes it from the case verdict entirely. |
| `params` | mapping | no | Evaluator-specific parameters, validated against that evaluator's own Pydantic params model at load time. |

## Defaults-merge rule

`params` merges field-by-field: a case's `GenerationParams` only overrides the
fields it *explicitly* set (tracked via Pydantic's `model_fields_set`), so a
case that never mentions `temperature` inherits the suite default rather than
silently resetting it to `None`. The one exception is `extra` (vendor-specific
knobs with no neutral spelling), which is merged key-by-key rather than
replaced wholesale.

`evaluators` does **not** merge this way. A case that supplies its own
`evaluators` list replaces `defaults.evaluators` entirely for that case; a
case that supplies none inherits the suite default list unchanged. There is
no per-field merging of individual evaluator specs — if a case needs the
suite's default evaluators *plus* one more, its `evaluators` list must repeat
the defaults explicitly.

`tags` **do** merge: a case's tags are the union of `defaults.tags` and the
case's own `tags`, not a replacement.

## The multi-evaluator pass rule

A case can carry more than one evaluator. The case is recorded as passed
(`CaseResult.passed = True`) only when **every evaluator with `required:
true` passed**. An evaluator with `required: false` still runs and its
result is recorded and included in per-evaluator metrics, but it cannot by
itself fail the case. `on_error` controls how an evaluator's own `ERROR`
status is folded into that rule for a `required` evaluator: `error`
propagates the case to an error-influenced state, `fail` counts it as a
failure, `skip` excludes that evaluator from the case verdict as if it were
absent for this case.

## Category vs. tags

`category` is a single optional classification per case (`geography`,
`arithmetic`, …), used for `AggregateMetrics.by_category` and for the
per-category regression gate (`category.*.pass_rate` in the shipped
threshold policy — evaluated only for categories with at least the
configured `min_samples` cases in *both* compared runs). `tags` are zero or
more free-form labels used for `AggregateMetrics.by_tag`, for `llm-eval run
--select-tag` case selection, and for filtering `llm-eval show --cases`. Use
`category` when a case belongs to exactly one bucket you want reported and
gated on; use `tags` for anything a case might carry more than one of, or
that exists purely for filtering rather than reporting.

## Hashing

Two content hashes exist so a run stays comparable and interpretable
independent of a suite file's later history.

**`case_hash`** (`compute_case_hash` in `datasets/hashing.py`) is computed
over exactly: `id`, `messages` (the case's `input` normalized into message
form), `system`, `expected`, `params` and `evaluators`. It **deliberately
excludes** `tags`, `category`, `metadata` and `weight`. Those four fields
describe how a case is *filed* — its label, its bucket, free-form notes — not
what it asks the model or how it is judged. Including them in the hash would
mean that re-tagging or re-categorizing a case (a purely organizational edit)
breaks historical pairing between runs, which is exactly the kind of drift
the hash exists to prevent.

**`suite_hash`** (`compute_suite_hash`) is computed over `schema_version`,
`name`, `version`, and each case's `id` and `case_hash`, sorted by `id` so
reordering cases in the source file does not change the digest. Because it
is built from case hashes rather than raw case content, it inherits the same
exclusions for free.

Both are content digests: two suites with byte-identical hashed content
produce the same hash regardless of file path, formatting, or key order. This
is what lets `llm-eval compare` decide whether two runs used the "same"
suite (`RegressionReport.suite_hash_match`) and what lets a `ResolvedCase`
from one run be paired with the "same" case in another run purely by hash,
even after the source file changed or moved.
