/**
 * Baseline against candidate: the gate's answer and everything behind it.
 *
 * The verdict is the backend's, computed from point estimates so that the same
 * two runs always produce the same word. Intervals, McNemar p-values and
 * overlap flags sit beside the verdict as context and never change it. Where a
 * check did not have enough samples to be meaningful, the row says so with a
 * badge instead of printing a difference the sample size cannot support.
 */

import { useSearchParams } from "react-router-dom";

import type { CheckOutcome, CheckStatus } from "../api/client";
import { useComparison, useRunMetrics } from "../api/queries";
import { Badge, Caveat, type BadgeTone } from "../components/Badge";
import { ConfidenceInterval } from "../components/ConfidenceInterval";
import { DeltaBadge } from "../components/DeltaBadge";
import { EmptyState } from "../components/EmptyState";
import { Loading, QueryError } from "../components/QueryState";
import { ScrollRegion } from "../components/ScrollRegion";
import { VerdictBanner } from "../components/VerdictBanner";
import {
  EMPTY,
  type Interval,
  formatCost,
  formatCount,
  formatInterval,
  formatLatency,
  formatNumber,
  formatPValue,
  formatPercent,
  formatScore,
  intervalsOverlap,
  shortHash,
} from "../lib/format";

const CHECK_TONE: Record<CheckStatus, BadgeTone> = {
  passed: "pass",
  failed: "fail",
  warning: "warn",
  insufficient_data: "warn",
  missing_metric: "muted",
};

/** How a metric's value should read, inferred from its dotted path. */
function kindOf(metric: string): "rate" | "latency" | "cost" | "score" {
  if (metric.endsWith("_ms")) return "latency";
  if (metric.includes("cost")) return "cost";
  if (metric.endsWith("rate") || metric.endsWith("coverage")) return "rate";
  return "score";
}

function formatMetricValue(metric: string, value: number | null): string {
  if (value === null || !Number.isFinite(value)) return EMPTY;
  switch (kindOf(metric)) {
    case "latency":
      return formatLatency(value);
    case "cost":
      return formatCost(String(value));
    case "rate":
      return formatPercent(value);
    case "score":
      return formatScore(value);
  }
}

function deltaUnit(metric: string): "pp" | "ms" | "raw" {
  const kind = kindOf(metric);
  if (kind === "rate") return "pp";
  if (kind === "latency") return "ms";
  return "raw";
}

function describeThreshold(check: CheckOutcome): string {
  if (check.threshold_value === null) return "no bound";
  const bound = check.violated_bound ?? "bound";
  return `${bound} ${formatNumber(check.threshold_value, 4)}`;
}

/**
 * The per-side interval on a check's baseline or candidate value.
 *
 * A rate is never shown bare, so when the backend supplied no interval for one
 * side — which it does for every rate that is not a paired pass rate — the cell
 * says so rather than leaving the number unqualified. Non-rate metrics such as
 * a latency percentile have no interval by design and are left unannotated.
 */
function SideInterval({
  metric,
  interval,
}: {
  metric: string;
  interval: Interval | null | undefined;
}) {
  const formatted = formatInterval(interval);
  if (formatted !== null) return <span className="ci">{formatted}</span>;
  if (kindOf(metric) !== "rate") return null;
  return (
    <span
      className="ci-missing"
      title="The backend computed no interval for this side. In paired mode the interval it does compute is the one on the delta, shown to the right."
    >
      (no interval)
    </span>
  );
}

function CaseIdList({ ids, emptyText }: { ids: readonly string[]; emptyText: string }) {
  if (ids.length === 0) return <p className="footnote">{emptyText}</p>;
  return (
    <ul className="case-list">
      {ids.map((id) => (
        <li key={id}>{id}</li>
      ))}
    </ul>
  );
}

export function Comparison() {
  const [params] = useSearchParams();
  const baseline = params.get("baseline");
  const candidate = params.get("candidate");

  const report = useComparison(baseline, candidate);
  const baselineMetrics = useRunMetrics(baseline ?? undefined, {
    enabled: Boolean(baseline),
  });
  const candidateMetrics = useRunMetrics(candidate ?? undefined, {
    enabled: Boolean(candidate),
  });

  if (baseline === null || candidate === null) {
    return (
      <>
        <div className="page-head">
          <h1>Comparison</h1>
        </div>
        <div className="panel">
          <EmptyState title="No runs selected">
            This view compares two runs by id. Open it with both, as in
            <code> /compare?baseline=&lt;run-id&gt;&amp;candidate=&lt;run-id&gt;</code>, or pick
            a pair from the Runs list.
          </EmptyState>
        </div>
      </>
    );
  }

  if (report.isPending) return <Loading what="the comparison" />;
  if (report.isError) return <QueryError what="the comparison" error={report.error} />;

  const data = report.data;
  const baseAgg = baselineMetrics.data?.metrics;
  const candAgg = candidateMetrics.data?.metrics;

  const categories = new Set([
    ...Object.keys(baseAgg?.by_category ?? {}),
    ...Object.keys(candAgg?.by_category ?? {}),
  ]);

  return (
    <>
      <div className="page-head">
        <h1>Comparison</h1>
        <span className="mono dim">
          {data.baseline_label ?? shortHash(data.baseline_run_id)} to{" "}
          {data.candidate_label ?? shortHash(data.candidate_run_id)}
        </span>
      </div>
      <p className="page-sub">
        Thresholds from {data.thresholds_id} v{data.thresholds_version}. Bounds compare point
        estimates; every interval on this page is advisory context.
      </p>

      <VerdictBanner report={data} />

      <div className="panel">
        <div className="panel-head">
          <h2>Checks</h2>
          <span className="dim">
            baseline, candidate, delta, threshold and the interval around the delta
          </span>
        </div>
        {data.checks.length === 0 ? (
          <EmptyState title="No checks were evaluated">
            The threshold policy named no metric these two runs both report.
          </EmptyState>
        ) : (
          <ScrollRegion label="Regression checks table">
            <table className="data">
              <caption className="sr-only">
                One row per threshold check, with the baseline and candidate values, the
                delta, the interval on the delta, the bound, the sample counts and the
                advisory p-value.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Check</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="num">
                    Baseline
                  </th>
                  <th scope="col" className="num">
                    Candidate
                  </th>
                  <th scope="col" className="num">
                    Delta
                  </th>
                  <th scope="col">Interval on the delta</th>
                  <th scope="col">Threshold</th>
                  <th scope="col" className="num">
                    n base / cand
                  </th>
                  <th scope="col" className="num">
                    p
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.checks.map((check) => {
                  const insufficient = check.status === "insufficient_data";
                  const missing = check.status === "missing_metric";
                  const overlap =
                    check.intervals_overlap ??
                    intervalsOverlap(check.baseline_ci, check.candidate_ci);
                  return (
                    <tr key={`${check.metric}-${check.label}`}>
                      <td>
                        <div>{check.label}</div>
                        <div className="mono dim">{check.metric}</div>
                      </td>
                      <td>
                        <Badge tone={CHECK_TONE[check.status]}>
                          {check.status.replace("_", " ")}
                        </Badge>
                      </td>
                      {insufficient || missing ? (
                        <>
                          <td className="num dim">
                            {missing ? "not reported" : "too few samples"}
                          </td>
                          <td className="num dim">
                            {missing ? "not reported" : "too few samples"}
                          </td>
                          <td className="num">
                            <Badge
                              tone={missing ? "muted" : "warn"}
                              title={
                                check.note ??
                                "This check did not reach its minimum sample count, so no difference is reported."
                              }
                            >
                              {missing ? "missing metric" : "insufficient samples"}
                            </Badge>
                          </td>
                        </>
                      ) : (
                        <>
                          <td className="num">
                            {formatMetricValue(check.metric, check.baseline)}
                            <SideInterval metric={check.metric} interval={check.baseline_ci} />
                          </td>
                          <td className="num">
                            {formatMetricValue(check.metric, check.candidate)}
                            <SideInterval metric={check.metric} interval={check.candidate_ci} />
                          </td>
                          <td className="num">
                            <DeltaBadge
                              delta={check.delta}
                              direction={check.direction}
                              unit={deltaUnit(check.metric)}
                            />
                            {check.relative_delta === null ? null : (
                              <div className="dim">
                                {formatPercent(check.relative_delta, 1)} relative
                              </div>
                            )}
                          </td>
                        </>
                      )}
                      <td>
                        {check.confidence_interval ? (
                          <span className="ci">
                            {formatInterval(check.confidence_interval)}
                          </span>
                        ) : (
                          <span className="ci-missing">(no interval)</span>
                        )}
                        {overlap === true ? (
                          <Caveat title="The baseline and candidate intervals overlap. The gate verdict is unchanged; the data simply does not separate them.">
                            inconclusive: intervals overlap
                          </Caveat>
                        ) : null}
                      </td>
                      <td className="mono">{describeThreshold(check)}</td>
                      <td className="num">
                        {formatCount(check.n_baseline)} / {formatCount(check.n_candidate)}
                        {check.n_paired === null ? null : (
                          <div className="dim">{formatCount(check.n_paired)} paired</div>
                        )}
                      </td>
                      <td className="num">
                        {formatPValue(check.p_value)}
                        {check.significant === true ? (
                          <div className="dim">advisory only</div>
                        ) : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </ScrollRegion>
        )}
        <p className="panel-note">
          A benchmark of a few dozen cases cannot resolve a difference smaller than roughly
          25 to 40 percentage points. A threshold breach below that size is a policy
          decision, not a demonstrated regression, which is why the p-value and the
          intervals are printed beside every bound.
        </p>
      </div>

      <div className="grid grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Regressions</h2>
            <span className="dim">
              {formatCount(data.summary.newly_failing.length)} cases newly failing
            </span>
          </div>
          <div className="panel-body">
            <CaseIdList
              ids={data.summary.newly_failing}
              emptyText="No case that passed in the baseline fails in the candidate."
            />
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Improvements</h2>
            <span className="dim">
              {formatCount(data.summary.newly_passing.length)} cases newly passing
            </span>
          </div>
          <div className="panel-body">
            <CaseIdList
              ids={data.summary.newly_passing}
              emptyText="No case that failed in the baseline passes in the candidate."
            />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Largest score drops</h2>
        </div>
        {data.summary.largest_score_drops.length === 0 ? (
          <EmptyState title="No per-case score dropped">
            Either no case is scored on a continuous scale, or no scored case moved down.
          </EmptyState>
        ) : (
          <ScrollRegion label="Largest score drops table">
            <table className="data">
              <caption className="sr-only">
                Cases whose score fell most between the two runs.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Case</th>
                  <th scope="col">Category</th>
                  <th scope="col" className="num">
                    Baseline
                  </th>
                  <th scope="col" className="num">
                    Candidate
                  </th>
                  <th scope="col" className="num">
                    Delta
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.summary.largest_score_drops.map((row) => (
                  <tr key={row.case_id}>
                    <td className="mono">{row.case_id}</td>
                    <td className="mono">{row.category ?? EMPTY}</td>
                    <td className="num">{formatScore(row.baseline_score)}</td>
                    <td className="num">{formatScore(row.candidate_score)}</td>
                    <td className="num">
                      <DeltaBadge delta={row.delta} direction="higher_is_better" unit="raw" digits={3} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollRegion>
        )}
      </div>

      <div className="grid grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Latency change</h2>
          </div>
          {baseAgg === undefined || candAgg === undefined ? (
            <Loading what="both runs' latency statistics" />
          ) : (
            <ScrollRegion label="Latency change table">
              <table className="data">
                <caption className="sr-only">
                  Latency percentiles in both runs, their difference, and whether either run
                  was below the sample floor.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Percentile</th>
                    <th scope="col" className="num">
                      Baseline
                    </th>
                    <th scope="col" className="num">
                      Candidate
                    </th>
                    <th scope="col" className="num">
                      Delta
                    </th>
                    <th scope="col">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(["p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"] as const).map((key) => {
                    const base = baseAgg.latency[key];
                    const cand = candAgg.latency[key];
                    const low =
                      baseAgg.latency.low_confidence.includes(key) ||
                      candAgg.latency.low_confidence.includes(key);
                    return (
                      <tr key={key}>
                        <td className="mono">{key.replace("_ms", "")}</td>
                        <td className="num">{formatLatency(base)}</td>
                        <td className="num">{formatLatency(cand)}</td>
                        <td className="num">
                          <DeltaBadge
                            delta={base === null || cand === null ? null : cand - base}
                            direction="lower_is_better"
                            unit="ms"
                          />
                        </td>
                        <td>
                          {low ? (
                            <Badge tone="warn">insufficient samples</Badge>
                          ) : (
                            <span className="dim">
                              n {formatCount(baseAgg.latency.n)} / {formatCount(candAgg.latency.n)}
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </ScrollRegion>
          )}
          <p className="panel-note">
            Latency deltas are point differences between percentiles. No interval is drawn
            around them: a bootstrap on percentile differences is an explicit opt-in in this
            project, not a default.
          </p>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Cost change</h2>
          </div>
          {baseAgg === undefined || candAgg === undefined ? (
            <Loading what="both runs' cost figures" />
          ) : (
            <ScrollRegion label="Cost change table">
              <table className="data">
                <caption className="sr-only">
                  Cost totals, per-case cost, price coverage and usage gaps in both runs.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Figure</th>
                    <th scope="col" className="num">
                      Baseline
                    </th>
                    <th scope="col" className="num">
                      Candidate
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Total</td>
                    <td className="num">{formatCost(baseAgg.cost.total_cost)}</td>
                    <td className="num">{formatCost(candAgg.cost.total_cost)}</td>
                  </tr>
                  <tr>
                    <td>Per case</td>
                    <td className="num">{formatCost(baseAgg.cost_per_case)}</td>
                    <td className="num">{formatCost(candAgg.cost_per_case)}</td>
                  </tr>
                  <tr>
                    <td>Price coverage</td>
                    <td className="num">{formatPercent(baseAgg.cost_coverage)}</td>
                    <td className="num">{formatPercent(candAgg.cost_coverage)}</td>
                  </tr>
                  <tr>
                    <td>Cases without usage</td>
                    <td className="num">{formatCount(baseAgg.tokens.n_missing_usage)}</td>
                    <td className="num">{formatCount(candAgg.tokens.n_missing_usage)}</td>
                  </tr>
                </tbody>
              </table>
            </ScrollRegion>
          )}
          <p className="panel-note">
            Coverage below 100% means some cases carried no price. Those cases are excluded
            from the totals above; they are never counted as costing nothing.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Affected categories</h2>
        </div>
        {categories.size === 0 ? (
          <EmptyState title="No categories recorded">
            Neither run tagged its cases with a category, so there is nothing to break down.
          </EmptyState>
        ) : (
          <ScrollRegion label="Affected categories table">
            <table className="data">
              <caption className="sr-only">
                Pass rate with its interval in each run, per category, with the difference.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Category</th>
                  <th scope="col">Baseline pass rate</th>
                  <th scope="col">Candidate pass rate</th>
                  <th scope="col" className="num">
                    Delta
                  </th>
                </tr>
              </thead>
              <tbody>
                {[...categories].sort().map((name) => {
                  const base = baseAgg?.by_category[name];
                  const cand = candAgg?.by_category[name];
                  const delta =
                    base?.pass_rate === null ||
                    base?.pass_rate === undefined ||
                    cand?.pass_rate === null ||
                    cand?.pass_rate === undefined
                      ? null
                      : cand.pass_rate - base.pass_rate;
                  return (
                    <tr key={name}>
                      <td className="mono">{name}</td>
                      <td>
                        <ConfidenceInterval
                          rate={base?.pass_rate ?? null}
                          interval={base?.pass_rate_ci ?? null}
                          n={base?.n ?? null}
                        />
                      </td>
                      <td>
                        <ConfidenceInterval
                          rate={cand?.pass_rate ?? null}
                          interval={cand?.pass_rate_ci ?? null}
                          n={cand?.n ?? null}
                          compareWith={base?.pass_rate_ci ?? null}
                        />
                      </td>
                      <td className="num">
                        <DeltaBadge delta={delta} direction="higher_is_better" unit="pp" />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </ScrollRegion>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Case set differences</h2>
          <span className="dim">
            suite hashes {data.suite_hash_match ? "match" : "differ"}
          </span>
        </div>
        <div className="panel-body grid grid-2">
          <div>
            <h3>Changed case hash ({formatCount(data.changed_cases.length)})</h3>
            <CaseIdList
              ids={data.changed_cases}
              emptyText="Every paired case has identical content in both runs."
            />
          </div>
          <div>
            <h3>Only in baseline ({formatCount(data.only_in_baseline.length)})</h3>
            <CaseIdList ids={data.only_in_baseline} emptyText="None." />
          </div>
          <div>
            <h3>Only in candidate ({formatCount(data.only_in_candidate.length)})</h3>
            <CaseIdList ids={data.only_in_candidate} emptyText="None." />
          </div>
          <div>
            <h3>Still failing ({formatCount(data.summary.still_failing.length)})</h3>
            <CaseIdList
              ids={data.summary.still_failing}
              emptyText="No case fails in both runs."
            />
          </div>
        </div>
        <p className="panel-note">
          A changed case hash means the case content itself moved between the two runs, so a
          difference in its result is not necessarily a model difference.
        </p>
      </div>
    </>
  );
}
