# Evaluators

An evaluator scores one model response against one resolved case. Ten types
ship with `llm-eval-lab`: five deterministic (no model call, no network, no
randomness), two structural JSON checks, two similarity measures, and one
LLM-as-judge evaluator. `uv run llm-eval evaluators --json` lists exactly
these ten, each with its parameter JSON Schema when `--schema` is passed. A
`BenchmarkCase` references one by its `type` string plus a `params` mapping
validated against that evaluator's own parameter model at benchmark load
time — an unknown type or a parameter that fails validation is a benchmark
validation error naming the case and the field, never a run-time surprise.

Every evaluator distinguishes a **failed** evaluation (the model's answer was
wrong; the evaluator reached a conclusion) from an **errored** one (the
evaluator itself could not reach a conclusion — a malformed regex timeout, a
missing embedding backend, an unparseable judge response). The two are never
conflated: `EvaluationResult.status` reports which happened, `.passed`
reports the case-level verdict, and a validator on the model itself refuses
any pairing that contradicts that distinction.

## The five deterministic evaluators

No model call, no network access, no randomness. Given the same response and
the same parameters, each returns the same verdict on every machine, every
time — which is what makes them usable as regression gates without any
statistical caveat about model nondeterminism.

### `exact_match` / `exact_match_ci`

String equality, case-sensitive (`exact_match`) or case-insensitive via
`str.casefold` (`exact_match_ci`), after normalization.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `expected` | string \| null | `null` | Overrides the case's `expected` field for this evaluator only. |
| `strip` | bool | `true` | Strip leading/trailing whitespace. |
| `normalize_whitespace` | bool | `false` | Collapse every whitespace run to one space, applied before `strip`. |

```yaml
evaluators:
  - type: exact_match
    params:
      strip: true
      normalize_whitespace: true
```

Worked example (`examples/benchmarks/exact_match_qa.yaml`, case
`whitespace-tolerant`): expected `"  hello  "`, response `"hello"` — both
normalize to `"hello"` and the case passes.

### `contains`

Substring presence, scored as the fraction of configured values found.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `values` | list of strings, ≥1 | required | Substrings to look for. |
| `mode` | `"all"` \| `"any"` | `"all"` | `all` requires every value present; `any` requires at least one. |
| `case_sensitive` | bool | `true` | |

The score is always `n_found / len(values)` regardless of `mode`, so a
response matching two of three required strings is visible in metrics even
when it fails. `contains` only checks for **presence** — to require that a
string is *absent*, use `regex` with `must_match: false` instead (see below);
`contains` has no polarity parameter for this.

```yaml
evaluators:
  - type: contains
    params:
      values: ["100", "Celsius"]
      mode: all
```

### `regex`

Pattern matching against the response text, under a wall-clock bound. The
pattern is compiled and length-capped at benchmark **load** time (an
uncompilable pattern is a validation error naming the case), and every match
runs in a **separate process** under a timeout — not a thread, because
CPython's `re` engine holds the GIL for the duration of a match, so a
catastrophic-backtracking pattern on a thread would stall the whole event
loop rather than merely time out. On expiry the worker is killed and the
evaluation is `status=ERROR` with `metadata["error_code"] = "regex_timeout"`.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pattern` | string, ≤1000 chars | required | Compiled at load time. |
| `must_match` | bool | `true` | `true` requires a match; `false` requires its absence. |
| `group` | int \| null | `null` | Capture group recorded in `metadata["captured"]`. |
| `ignore_case` | bool | `false` | |
| `multiline` | bool | `false` | |
| `dotall` | bool | `false` | |
| `fullmatch` | bool | `false` | Anchor to the whole string instead of searching within it. |
| `match_timeout_s` | float | `2.0` | Further capped by `settings.max_regex_match_s`. |

```yaml
evaluators:
  - type: regex
    id: no-sydney-mentioned
    params:
      pattern: "(?i)sydney"
      must_match: false
```

### `numeric_tolerance`

Extracts a number from the response and compares it to `expected` (or a
`params.expected` override) under `math.isclose` semantics: the case passes
when **either** configured tolerance is satisfied; with neither set, the
comparison is exact. A response with no extractable number is a **failed**
evaluation, not an error — the evaluator worked; the answer just wasn't a
number.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `expected` | float \| null | `null` | Overrides the case's `expected` field. |
| `abs_tol` | float ≥0 \| null | `null` | Absolute tolerance. |
| `rel_tol` | float ≥0 \| null | `null` | Relative tolerance. |
| `extract` | `"first"` \| `"last"` \| `"only"` | `"first"` | Which number in the response to compare; `only` requires exactly one number to be present. |

```yaml
evaluators:
  - type: numeric_tolerance
    params:
      abs_tol: 0.01
```

Worked example (`examples/benchmarks/numeric_reasoning.yaml`): expected
`3.14`, `abs_tol: 0.01` — a response of `"pi is approximately 3.14159..."`
extracts `3.14159` and passes, since `|3.14159 - 3.14| < 0.01`.

## The two structural JSON evaluators

Both answer a question about the **shape** of a response, both are fully
deterministic, and both treat a malformed response as a failed evaluation
(the model answered wrongly), never an error. A single wrapping markdown
code fence is stripped before parsing when `allow_markdown_fence` is true
(the default) — a response containing two fenced blocks is not unwrapped, on
the principle that guessing which block is the answer is how an evaluator
starts inventing results.

### `json_valid`

Whether the response parses as JSON, optionally requiring a top-level
object.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `require_object` | bool | `false` | Require the top-level value to be a JSON object rather than any JSON value. |
| `allow_markdown_fence` | bool | `true` | Strip one wrapping ```` ``` ````-fence before parsing. |

```yaml
evaluators:
  - type: json_valid
    params:
      require_object: true
```

### `json_schema`

Whether the response satisfies a JSON Schema, with **partial credit**:
`score = 1 / (n_errors + 1)`, so a response with one violation scores `0.5`
and one with three scores `0.25` — monotonically decreasing, never zero,
never negative, and every violation is listed in `metadata["errors"]` so the
number is never the only evidence.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `schema` (YAML key; field name `json_schema`) | JSON Schema object | required | Remote `$ref` (anything not a local `#/...` pointer) is refused at benchmark load time — a benchmark file must never make validation issue an outbound request. Nesting depth and byte size are also capped at load time. |
| `allow_markdown_fence` | bool | `true` | |
| `validation_timeout_s` | float | `2.0` | Applies only when the schema asserts a `pattern`/`patternProperties` keyword. |

A schema asserting `pattern` or `patternProperties` — a regular expression
supplied by whoever wrote the benchmark, matched against model output — is
validated in a separate process under a wall-clock bound, through the same
mechanism the `regex` evaluator uses, for the identical reason: an
unbounded regex match holds the GIL and can stall the whole run. A schema
with no such keyword (the overwhelming majority) validates in-process, since
there is no untrusted regex to bound. Format assertions (`format: email`,
`format: date-time`, …) stay off, matching `jsonschema`'s own default — a
`format` keyword is an annotation here, never an assertion, so the same
benchmark scores identically regardless of which optional format libraries
happen to be installed.

```yaml
evaluators:
  - type: json_schema
    params:
      schema:
        type: object
        required: [name, age]
        properties:
          name: {type: string}
          age: {type: integer, minimum: 0}
```

Worked example: a response `{"name": "Ada"}` (missing `age`) against the
schema above has one violation and scores `0.5`; `{"name": "Ada", "age":
"thirty"}` (wrong type) also scores `0.5` for its own violation;
`{"name": 1, "age": "x"}` has two violations and scores `1/3`.

## The two similarity evaluators, kept deliberately apart

`lexical_similarity` and `semantic_similarity` answer different questions —
surface word overlap versus semantic closeness — and by design have
different names, different parameters, and **no code path between them**.
There is no fallback from one to the other: a `semantic_similarity` case
with no `embedding_provider` is a **benchmark validation error at load
time**, naming the case and the field, never a silent downgrade to lexical
overlap. Silently substituting word overlap for meaning is exactly the kind
of false precision this project exists to avoid — the number would still be
printed and the dashboard would still render it, with nobody reading the
report knowing the question had quietly changed. Every result from either
evaluator carries `metadata["method"]` (`"lexical:token_set"`,
`"lexical:char_ngram"`, or `"embedding:<provider>:<model>"`), which the
dashboard renders, so a reader always knows which question was actually
answered.

### `lexical_similarity`

**This evaluator measures surface word overlap, not meaning.** It will
falsely fail a correct paraphrase that shares little vocabulary with the
reference, and it will falsely pass a wrong answer that happens to reuse the
reference's words. Treat it as a coarse, zero-dependency smoke check that
keeps an evaluation functional with no setup — never present its score in a
dashboard or report with the same visual confidence as an embedding-based or
judge-based score.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `expected` | string \| null | `null` | Overrides the case's `expected` field. |
| `threshold` | float [0,1] | `0.6` | Overlap at or above which the case passes. Coarse — calibrate per suite. |
| `method` | `"token_set"` \| `"char_ngram"` | `"token_set"` | `token_set` compares word **sets** (Jaccard); `char_ngram` compares character n-gram sets, which tolerates inflection and small typos. |
| `ngram` | int, 2–8 | `3` | Character n-gram length; used only by `method: char_ngram`. |
| `normalize` | bool | `true` | NFKC-normalize, case-fold, strip punctuation, collapse whitespace before comparing. |

```yaml
evaluators:
  - type: lexical_similarity
    params:
      threshold: 0.6
      method: token_set
```

**Threshold guidance:** roughly 0.5–0.6 as "similar" is a reasonable coarse
default for `token_set`. This is much lower-fidelity than an
embedding-based threshold and is explicitly labeled as such in the
evaluator's own `explanation` text on every result.

### `semantic_similarity`

Cosine similarity between provider embeddings of the response and the
reference. `embedding_provider` is a **required** parameter — a
`ProviderConfig` naming the provider and model that produce the embeddings
— with no default and no fallback.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `embedding_provider` | `ProviderConfig` | **required** | The embedding backend. Must implement the `EmbeddingProvider` capability (an `embed(texts) -> list[list[float]]` method); a provider that does not is a configuration-kind evaluation error naming the case, not a silent fallback. |
| `expected` | string \| null | `null` | Overrides the case's `expected` field. |
| `threshold` | float [-1,1] | `0.82` | Cosine at or above which the case passes. |
| `normalize` | bool | `true` | Same text normalization as `lexical_similarity`, applied before embedding. |

```yaml
evaluators:
  - type: semantic_similarity
    params:
      embedding_provider:
        provider: openai
        model: text-embedding-3-small
      threshold: 0.82
```

**Threshold guidance:** embedding cosine around 0.80–0.85 is a reasonable
"similar enough" default, but it **must be calibrated per embedding model
and never carried across a model change** — the `method` metadata tag is
exactly what makes such a change visible in stored results.

**Fake-provider note:** the deterministic fake provider used throughout this
repository's examples **does** implement the `EmbeddingProvider` capability,
so `semantic_similarity` can be exercised fully offline with
`embedding_provider: {provider: fake, model: fake-embed}`. Its vectors are
derived from a hash of the text (dimension 16, unit-normalized, seeded by the
provider's `seed` option) — reproducible across processes, but carrying no
actual semantic content: two unrelated texts embed to *different* vectors,
not to *dissimilar meaning*. This is enough to exercise the evaluator's own
plumbing (embedding, cosine, thresholding, the `method` tag) without a vendor
credential, which is what
`examples/benchmarks/similarity_paraphrase.yaml`'s
`semantic-similarity-shape-demo` case does. It is not a substitute for a real
embedding-capable provider such as `openai` when the question being asked is
actually about meaning.

## `llm_judge`: LLM-as-judge

Model-graded scoring against a weighted rubric. **A judge score is an
instrument reading, not ground truth**, and every design choice below
follows from that one sentence.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `provider` | `ProviderConfig` \| null | `null` | Single judge. Exactly one of `provider`/`judges` is required. |
| `judges` | tuple of `ProviderConfig` | `()` | A judge panel; scored the same way and aggregated further (see Panels, below). |
| `template_id` | string | `"single_answer_grading_v2"` | Versioned prompt template. |
| `rubric` | string | **required** | No default rubric — a judge run against an unstated standard produces numbers nobody can interpret. |
| `criteria` | list of `{id, description, weight}` | `()` | Weighted dimensions scored separately. Weights are normalized by their own sum; a rubric with none is graded as a whole under one implicit `overall` criterion. |
| `scale` | `{kind, minimum, maximum, labels}` | integer 1–5 | `labels` maps a scale value to an anchor description shown to the judge. |
| `use_reference` | bool | `true` | Whether the case's `expected` field (or a `reference` override) is shown to the judge as a known-good answer. |
| `reference` | string \| null | `null` | Overrides the case's `expected` field as the reference shown to the judge. |
| `pass_threshold` | float \| null | `null` | On the **raw** scale the rubric declares (e.g. `4` on a 1–5 scale), not the normalized 0–1 score. Out-of-range is a benchmark validation error. |
| `repetitions` | int, 1–9 | `1` | Repeat judgments per judge; reliability mode is `3`, opt-in. |
| `aggregation` | `"mean"` \| `"median"` \| `"majority"` \| `"min"` | `"median"` | How repeats for one judge are combined. |
| `params` | `GenerationParams` | `temperature=0.0, response_format=json_object` | Judge decoding parameters. |
| `max_output_chars` | int | `8000` | Candidate output is truncated to this before being shown to the judge. |
| `high_variance_threshold` | float | `1.0` | Spread (on the raw scale) above which a judgment is flagged high-variance. |

```yaml
evaluators:
  - type: llm_judge
    params:
      provider:
        provider: anthropic
        model: claude-sonnet-5
      rubric: >-
        Score how well the candidate answers the question, using the
        reference answer as a guide.
      criteria:
        - id: correctness
          description: Does the answer state the correct fact?
          weight: 0.6
        - id: completeness
          description: Does it address all parts of the question?
          weight: 0.4
      scale: {kind: integer, minimum: 1, maximum: 5}
      pass_threshold: 3
      repetitions: 1
```

### How a judgment is produced

A versioned prompt template renders the rubric, the weighted criteria with
anchors, the scale, the case's question, the reference answer (when
`use_reference`), and the candidate output — the last three enclosed inside
delimiter blocks whose tags are BLAKE2b digests of the block's own text (a
fixed point nobody can forge by guessing), with a system-level instruction
that everything between those markers is data, never instructions, no matter
what it claims. `template_id` and `rubric_hash` are recorded on every
judgment, so an edit to the rubric is visible as a provenance change rather
than a silent shift in what a score means.

The judge's raw response is parsed strictly: JSON after removing at most one
wrapping markdown fence, then validated against a structural schema, then
checked that every configured criterion appears exactly once and every score
is an integer within the declared scale. Out-of-range scores are **rejected,
not clamped** — a judge answering 9 on a 1–5 scale did not mean 5. On the
first failure, exactly **one** repair attempt is sent, restating the
requirement without changing the judgment. A second failure produces
`status=ERROR` with `EvaluationErrorInfo(kind="parse")`. **A failed judgment
is never scored as 0 and never as the scale midpoint** — either would be
indistinguishable from a real judgment in every aggregate this project
computes, and would corrupt the mean without anyone knowing it happened.
When criteria are configured, the overall score is **computed** as the
weight-normalized sum of the per-criterion scores — never trusted from the
judge's own `overall_score` field, which is recorded and compared but never
decides the number, because arithmetic is not something a language model
should be relied on for when the inputs are already in hand.

### `passed` is set only when `pass_threshold` is configured

Otherwise the judge contributes `score` and leaves `passed` as `None`, which
the aggregate counts under `n_score_only` and excludes from `pass_rate`.
This is what stops a suite from accidentally reading a judge's opinion as a
pass/fail fact when the judge never returned one.

### Repeats and panels

Repeats for one judge are aggregated by **median** — robust to a single wild
judgment with no outlier-detection logic required. A panel (`judges`, plural)
takes the **median of the judges' own medians**, and reports min/median/max
across judges, never a single blended number. `dispersion` (the range across
every judgment collected) and `agreement` are always reported alongside the
score; `high_judge_variance` is flagged when the range exceeds
`high_variance_threshold`.

### Bias controls

- **Temperature 0** by default for every judge call.
- **Self-preference**: when a configured judge's provider and model equal
  the candidate's, `self_preference_risk: true` is set on the judgment —
  visible, never silent, never blocking the run.
- **Verbosity**: candidate response length is logged beside the judge score,
  so verbosity-correlated inflation is visible in the data even though
  nothing here corrects for it algorithmically.
- **Prompt injection**: the candidate output, the question and the reference
  answer are each wrapped in their own digest-tagged delimiter block with an
  explicit system instruction to treat the enclosed content as data and
  ignore any instructions found inside it.

### Cost and provenance

Judge calls are bounded by their own semaphore (`judge_concurrency`, default
4, held below the candidate's `concurrency`), and their tokens and cost are
attributed to `judge_cost` — kept apart from candidate-model cost, never
folded in silently, so a "cheap" candidate run using repeated judging can
correctly show up as expensive in judge spend without hiding it inside the
candidate's own cost line. Persisted per judge call: the fully rendered
prompt exactly as sent, the raw response text, the parsed scores, the
judge's provider/model/temperature/max output tokens, a timestamp, the
repeat index, latency, and token usage — everything needed to audit or
reproduce a judgment without trusting the score alone.

### Documented limitations

Every one of the following is a real, unresolved limitation of LLM-as-judge
evaluation as a methodology, not specific to this implementation. They are
listed here so a reader treats a judge score with the skepticism it deserves.

- **Self-preference**: judges tend to score outputs from their own model
  family higher, inflating comparisons that include a same-family candidate.
- **Positional bias**: in pairwise comparisons, judges disproportionately
  favor whichever answer is shown first (or second), independent of actual
  quality.
- **Verbosity bias**: judges tend to rate longer, more elaborate answers as
  better even when correctness doesn't differ.
- **Nondeterminism**: judges can return different scores for the same input
  across calls, even at temperature 0, due to provider-side sampling and
  infrastructure variance.
- **Model dependence**: absolute scores and thresholds calibrated against
  one judge model are not portable to a different judge model without
  recalibration.

Pairwise judge mode with position swapping (`JudgeConfig.swap_positions`)
exists in the frozen contract but is not implemented in this version — only
single-answer grading against a rubric ships. Setting `swap_positions` has
no effect; a future pairwise mode would need position randomization to
mitigate positional bias directly rather than only naming it as a
limitation.

## The plugin trust model (D-CUSTOM)

`llm-eval-lab` also supports custom, out-of-tree evaluators through Python
entry points — but not through anything reachable from a benchmark file
itself. **A benchmark file can never carry code.** It is parsed with
`yaml.safe_load` only; there is no `eval`, no `exec`, no dotted import path,
no expression language, and no sandbox for benchmark-supplied logic. A
custom evaluator arrives exactly one way: as an installed Python
distribution advertising an entry point in the group
`llm_eval_lab.evaluators`, because installing a package is a trust act an
operator performs deliberately, and opening a YAML file someone emailed you
is not.

Four independent gates must all be open before any third-party evaluator
code runs:

1. **`plugins_enabled` is `false` by default.** A fresh install loads
   nothing extra, ever.
2. **Only exact names in `plugins_allowed` are loaded** — an empty
   allowlist (the default) means no plugin loads even with the feature
   turned on. Matching is exact: no globs, no prefixes.
3. **Loading happens once, at process start**, from a command that
   deliberately chose to call it. It is never triggered by a request.
4. **Nothing under `api/` calls the plugin loader, ever.** The HTTP surface
   is reachable by anything that can reach the port, and a plugin is
   arbitrary third-party code — an API endpoint that could trigger plugin
   loading would turn "the operator installed and enabled a package
   locally" into "anyone who can reach this port can run code," which this
   project refuses categorically. A test asserts the absence of any such
   call path and that importing the API package does not even import the
   plugin loader.

What loaded is always recorded: `RunConfig.plugins` carries one
`PluginRecord` per loaded plugin — its entry-point name, its distribution
name and version, and the evaluator types it registered — so a run whose
scores depended on third-party code says so in its own persisted
configuration, never leaving a reader to infer it. A plugin cannot shadow a
built-in evaluator type; the registry refuses a name that is already taken.
