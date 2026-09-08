# Regression testing

`llm-eval compare BASELINE_RUN_ID CANDIDATE_RUN_ID` is the command a CI
pipeline gates on. It loads two stored runs, pairs their cases by
`case_hash` where they overlap, evaluates a threshold policy against the
result, and communicates its verdict through its exit code.

## Threshold file format

A threshold policy is a YAML or JSON document validated against
`llm_eval_lab.models.RegressionThresholds`. The package ships a default at
`src/llm_eval_lab/reporting/data/thresholds.yaml`; a copy usable outside an
installed wheel lives at `examples/thresholds/ci-thresholds.yaml`, and
`examples/thresholds/strict-thresholds.yaml` demonstrates a tighter policy
for a project that wants zero tolerance on regressions.

```yaml
id: ci
version: "1"
paired: true
require_same_suite: true
min_paired_cases: 1
fail_on_missing_metric: true
confidence_level: 0.95
checks:
  - metric: pass_rate
    label: overall pass rate
    direction: higher_is_better
    max_absolute_decrease: 0.02
    min_samples: 1
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | A name for the policy, echoed in reports as `thresholds_id`. |
| `version` | string | A version for the policy, echoed as `thresholds_version`. |
| `checks` | list of `MetricCheck`, at least one | The bounds evaluated. |
| `paired` | bool (default `true`) | Whether to attempt case-id pairing between the two runs. |
| `require_same_suite` | bool (default `true`) | When true, comparing runs of different suites (by `suite_hash`) is refused as `incomparable` rather than silently compared anyway. |
| `min_paired_cases` | int (default `1`) | Below this many paired cases, the comparison is `incomparable`. |
| `fail_on_missing_metric` | bool (default `true`) | A policy naming a metric neither run reports is a mismatch between the policy and the benchmark; this makes that a failure rather than a silent pass. |
| `confidence_level` | float (default `0.95`) | Confidence level used for every advisory Wilson/Method-10 interval the engine attaches to a check outcome. |

## Every `MetricCheck` bound

| Field | Type | Semantics |
|---|---|---|
| `metric` | string | A dotted metric path (`pass_rate`, `error_rate`, `latency.p95_ms`, `category.<name>.pass_rate`, `evaluator_type.<type>.pass_rate`, `evaluator.<id>.mean_score`, …). `*` matches every value at that position, producing one outcome per match (e.g. `category.*.pass_rate` yields one check per category both runs report). |
| `direction` | `higher_is_better` \| `lower_is_better` | **Descriptive only.** It labels the metric for rendering and picks the arrow a report draws. The bounds below are already directional; the engine derives no second directional rule from this field. |
| `label` | string \| null | Human-readable label for reports; defaults to the metric path. |
| `min_value` | float \| null | **Inclusive floor.** Violated when the candidate is *strictly below* it — a candidate exactly equal to `min_value` passes. |
| `max_value` | float \| null | **Exclusive ceiling.** Violated when the candidate is *greater than or equal to* it — a candidate exactly equal to `max_value` **fails**. This asymmetry is what makes "error rate must remain below 1%" expressible as `max_value: 0.01`. |
| `max_absolute_decrease` | float \| null | Violated when `candidate - baseline < -value`. |
| `max_relative_decrease` | float \| null | Violated when the candidate falls more than this fraction below the baseline. |
| `max_absolute_increase` | float \| null | Violated when `candidate - baseline > value`. |
| `max_relative_increase` | float \| null | Violated when the candidate exceeds `baseline * (1 + value)`. |
| `require_significant` | bool (default `false`) | **Reserved and unimplemented.** Setting it to `true` raises `EvaluatorConfigError` at policy-load time (exit code 3). It exists in the frozen contract for forward compatibility only. |
| `severity` | `"error" \| "warning"` (default `"error"`) | An `error`-severity breach drives the overall verdict to `fail` (exit 4). A `warning`-severity breach is reported but only affects the exit code when the caller passes `--fail-on-warning` (exit 7). |
| `min_samples` | int (default `1`) | A **gate**, not a filter. Below this many samples in either run, the check reports `INSUFFICIENT_DATA` rather than being silently skipped or silently passed — a sample count a run did not record counts as zero, so a gate is never bypassed by accident. |

## The shipped default policy

`src/llm_eval_lab/reporting/data/thresholds.yaml` (mirrored in
`examples/thresholds/ci-thresholds.yaml`) is used whenever `--thresholds` is
omitted. It checks:

| Metric | Bound | `min_samples` |
|---|---|---|
| `pass_rate` | no more than 2 percentage points worse | 1 |
| `category.*.pass_rate` | no more than 2 percentage points worse | 10 |
| `evaluator_type.json_valid.pass_rate` | zero tolerance — any decrease at all fails | 1 |
| `evaluator_type.json_schema.pass_rate` | zero tolerance | 1 |
| `latency.p95_ms` | no more than 15% relative increase | 20 |
| `error_rate` | must stay below 1% (exclusive ceiling) | 1 |
| `evaluator.*.mean_score` | no more than 0.05 absolute decrease (normalized 0–1 scale) | 1 |

JSON adherence gets zero tolerance because it is treated as near-deterministic
syntactic behavior, not model quality subject to normal noise. The latency
gate uses successful-attempt latency, not wall-clock time including retries,
so it measures the model rather than a rate limit — and 20 samples is the
point below which a p95 estimate sits within the top one or two values in the
sample, essentially reporting the near-maximum rather than a real p95 (see
"honesty about small samples," below).

## The exit-code contract

```
0   no regression (verdict pass, or verdict warn without --fail-on-warning)
2   one of the two run ids does not exist
4   a regression was detected (verdict fail)
6   the two runs are not comparable
7   warning-severity breaches only, with --fail-on-warning
```

A CI snippet:

```bash
uv run llm-eval compare "$BASELINE_RUN_ID" "$CANDIDATE_RUN_ID" \
  --thresholds examples/thresholds/ci-thresholds.yaml \
  --json > regression-report.json
status=$?
if [ "$status" -eq 4 ]; then
  echo "regression detected — see regression-report.json" >&2
  exit 1
elif [ "$status" -eq 6 ]; then
  echo "runs are not comparable — see regression-report.json" >&2
  exit 1
fi
exit 0
```

`--json` writes the `RegressionReport` document itself to stdout and nothing
else — not the usual command envelope every other `--json` command emits.
The report already carries its own `schema_version`, and it is byte-for-byte
the same document `GET /api/compare` returns and `llm-eval report --compare-to
... --format json` renders, so a pipeline consuming it does not need to know
which command produced the file. This guarantee holds on failure paths too:
an unknown run id or an unusable threshold policy writes **nothing** to
stdout and reports the problem on stderr instead, so a pipeline never finds a
malformed or partial document at the path it expected.

## Point-estimate gate semantics and the advisory interval

**Thresholds evaluate point estimates only.** This is a deliberate design
decision (`D-GATE`), not an oversight: a confidence-interval-based gate would
make the same two runs non-deterministically pass or fail across reruns
whenever a stochastic candidate happened to land near a boundary, which is
unacceptable for a CI gate that needs a stable, reproducible verdict for
identical inputs.

The confidence interval is still computed and attached to every applicable
check outcome — the Wilson score interval for each run's own pass rate, the
Newcombe Method-10 interval for the paired pass-rate delta, and the McNemar
exact two-sided p-value for the paired concordance pattern — but strictly as
**advisory context for a human**, never as part of the pass/fail logic. A
check that fails its point-estimate bound while its interval comfortably
straddles zero (or vice versa) carries a `note` saying so in the report; the
verdict itself never changes because of it. In **unpaired** mode (the two
runs don't share case ids — different benchmarks, or a benchmark that
changed), `CheckOutcome.baseline_ci` and `CheckOutcome.candidate_ci` carry
each run's own Wilson interval side by side instead of a single paired-delta
interval, and `intervals_overlap` reports whether they overlap; McNemar and
Method-10 are not computed without paired cases.

## The minimum-n gates

Two independent minimum-sample mechanisms exist, and both are gates, not
filters — below them the honest answer is "not enough data," not a
number computed anyway and not a check silently skipped.

- **Per-check `min_samples`** (above): a `MetricCheck` with fewer than this
  many samples in either run reports `INSUFFICIENT_DATA`.
- **Percentile minimum-n**, applied independently by the statistics layer
  before a latency percentile is even reported as a trustworthy number:

  | Percentile | Minimum n | Why |
  |---|---|---|
  | p50 (median) | 5 | The median is robust even at small n. |
  | p95 | 20 | Below this, the p95 rank sits within the top one or two values — effectively reporting the near-maximum. |
  | p99 | 100 | Below this, p99 collapses onto the sample maximum. |

  Below its gate, a percentile is reported with `LatencyStats.low_confidence`
  naming it, rather than a bare number that looks exactly as trustworthy as
  one computed over a real sample.

## Honesty about small samples

A 20–50 case benchmark **cannot reliably distinguish anything smaller than
roughly a 25–40 percentage-point pass-rate difference**, at 95% two-sided
significance with 80% power, for an independent two-proportion comparison.
Concretely (conservative upper bounds — paired designs with positively
correlated outcomes, the normal case here, are usually somewhat more
sensitive at the same n):

| n per arm | Minimum detectable difference @ baseline p=0.80 | @ p=0.90 | @ p=0.50 |
|---|---|---|---|
| 20 | 42.3 pp | 39.3 pp | 39.5 pp |
| 50 | 26.0 pp | 22.7 pp | 26.7 pp |
| 100 | 17.9 pp | 15.0 pp | 19.3 pp |
| 200 | 12.3 pp | 10.0 pp | 13.8 pp |
| 500 | 7.5 pp | 5.9 pp | 8.8 pp |

The 2-percentage-point default threshold on `pass_rate` above is a
**point-estimate gate for CI purposes**, not a claim that a 2pp difference is
statistically distinguishable from noise at the sample sizes most benchmark
suites in this repository use. A worked fixture makes this concrete: with
`n=50` paired cases, a candidate that drops from a 90% baseline pass rate to
84% (a 6pp drop — three times the default threshold) produces a McNemar exact
p-value of 0.45 and a Newcombe Method-10 interval on the delta of
approximately `(-0.175, 0.049)` — an interval that comfortably includes zero.
The point-estimate gate correctly fails this comparison against the default
2pp threshold; the advisory interval correctly says the underlying difference
is not distinguishable from noise at this sample size. Both statements are
true at once, and the report shows both rather than picking one. Treat every
threshold breach on a small suite as "this specific number crossed this
specific line," not as proof of a real regression — read the attached
interval and p-value before treating a CI failure as confirmed model
degradation, and grow the benchmark's case count for any comparison where
that distinction matters operationally.
