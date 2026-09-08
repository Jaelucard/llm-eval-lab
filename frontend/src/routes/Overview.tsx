/**
 * The landing view: what ran recently, which models lead, and what regressed.
 *
 * Every rate on this page is rendered through `ConfidenceInterval`, which
 * refuses to print a bare number: it shows the interval the API supplied, or
 * says none was supplied, and always shows the population. Nothing here
 * synthesises an interval the backend did not compute.
 *
 * The two headline tiles are deliberately conservative. The cost tile reports
 * how many models carried no price rather than treating them as free, and sums
 * the priced ones in fixed point rather than as floats. The latency tile
 * reports a median of per-run p95 values rather than pretending to a pooled
 * percentile the summary endpoint cannot give it.
 */

import { useQueries } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api, type CheckOutcome, type RegressionReport, type RunDetail } from "../api/client";
import { queryKeys, useModelSummary, useRuns } from "../api/queries";
import { Badge, Caveat } from "../components/Badge";
import { ConfidenceInterval } from "../components/ConfidenceInterval";
import { DeltaBadge } from "../components/DeltaBadge";
import { EmptyState, NoRunsYet } from "../components/EmptyState";
import { MetricTile } from "../components/MetricTile";
import { Loading, QueryError } from "../components/QueryState";
import { RunTable } from "../components/RunTable";
import { ScrollRegion } from "../components/ScrollRegion";
import {
  EMPTY,
  formatCost,
  formatCount,
  formatInterval,
  formatLatency,
  formatTimestamp,
  shortHash,
  sumDecimalStrings,
} from "../lib/format";

const RECENT_LIMIT = 10;
const BASELINE_SCAN = 6;

/** Median of a list of numbers, or null when the list is empty. */
function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid] ?? null;
  const low = sorted[mid - 1];
  const high = sorted[mid];
  return low === undefined || high === undefined ? null : (low + high) / 2;
}

/**
 * The pass-rate delta for one baselined run, with the interval the backend
 * computed for that delta and nothing invented beside it.
 *
 * The interval here is `CheckOutcome.confidence_interval` — the Newcombe
 * interval on the paired difference — and it is the only thing allowed to wear
 * the interval styling. A baseline-and-candidate pair is a pair of point
 * estimates, and rendering it in that styling would read as a range.
 */
function PassRateDelta({
  check,
  intervalsOverlapFlag,
}: {
  check: CheckOutcome | undefined;
  intervalsOverlapFlag: boolean | null | undefined;
}) {
  if (check === undefined) {
    return <span className="dim">{EMPTY}</span>;
  }
  if (check.status === "insufficient_data" || check.status === "missing_metric") {
    return (
      <Badge tone={check.status === "missing_metric" ? "muted" : "warn"} title={check.note ?? undefined}>
        {check.status === "missing_metric" ? "missing metric" : "insufficient samples"}
      </Badge>
    );
  }

  const ci = formatInterval(check.confidence_interval);
  const overlap = check.intervals_overlap ?? intervalsOverlapFlag;

  return (
    <span className="chip-row">
      <DeltaBadge delta={check.delta} direction={check.direction} unit="pp" />
      {ci === null ? (
        <span
          className="ci-missing"
          title="The backend computed no interval on this delta, which is what it reports for an unpaired comparison."
        >
          (no interval)
        </span>
      ) : (
        <span className="ci" title="95% Newcombe interval on the paired pass-rate difference.">
          {ci}
        </span>
      )}
      <span className="n-badge">
        n={check.n_paired ?? check.n_candidate}
        {check.n_paired === null ? " candidate" : " paired"}
      </span>
      {overlap === true ? (
        <Caveat title="The baseline and candidate intervals overlap, so this difference is not resolved at this sample size. The gate verdict is unchanged: it is computed from point estimates by design.">
          inconclusive: intervals overlap
        </Caveat>
      ) : null}
    </span>
  );
}

export function Overview() {
  const runs = useRuns({ limit: RECENT_LIMIT, order: "-created_at" });
  const models = useModelSummary();

  const recent = runs.data?.items ?? [];
  const scanIds = recent.slice(0, BASELINE_SCAN).map((run) => run.id);

  const details = useQueries({
    queries: scanIds.map((id) => ({
      queryKey: queryKeys.run(id),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.getRun(id, signal),
      staleTime: 15_000,
    })),
  });

  const withBaseline = details
    .map((query) => query.data)
    .filter((detail): detail is RunDetail => Boolean(detail?.run.baseline_run_id));

  const comparisons = useQueries({
    queries: withBaseline.map((detail) => {
      const baseline = detail.run.baseline_run_id ?? "";
      return {
        queryKey: queryKeys.compare(baseline, detail.run.id),
        queryFn: ({ signal }: { signal: AbortSignal }) =>
          api.compare(baseline, detail.run.id, signal),
        staleTime: 15_000,
        retry: false,
      };
    }),
  });

  const rows = models.data ?? [];
  const spend = sumDecimalStrings(rows.map((row) => row.total_cost));
  const runP95s = rows
    .map((row) => row.median_run_p95_ms ?? null)
    .filter((value): value is number => value !== null);
  const runsWithLatency = rows.reduce((sum, row) => sum + row.n_runs_with_latency, 0);

  return (
    <>
      <div className="page-head">
        <h1>Overview</h1>
      </div>
      <p className="page-sub">
        Recent runs, model standings and any run that declared a baseline. Every rate is shown
        with the interval the API computed for it, or with an explicit note that it computed
        none, and always with the population behind it.
      </p>

      {runs.isError ? <QueryError what="the runs listing" error={runs.error} /> : null}
      {models.isError ? <QueryError what="the model summary" error={models.error} /> : null}

      <div className="grid grid-tiles">
        <MetricTile
          label="Runs stored"
          value={formatCount(runs.data?.total ?? null)}
          sub={`${formatCount(recent.length)} shown below`}
        />
        <MetricTile
          label="Models evaluated"
          value={formatCount(rows.length)}
          sub={`${formatCount(rows.reduce((sum, row) => sum + row.n_runs, 0))} runs pooled`}
        />
        <MetricTile
          label="Priced spend"
          value={spend.total === null ? EMPTY : formatCost(spend.total)}
          sub={
            spend.skipped === 0
              ? `across ${formatCount(spend.counted)} models, all priced`
              : `${formatCount(spend.skipped)} of ${formatCount(rows.length)} models carried no price and are excluded, not counted as zero`
          }
          title="Unpriced models are never folded into a total. They are reported as a count. Priced amounts are summed in fixed point, not as floats."
        />
        <MetricTile
          label="Median run P95 latency"
          value={formatLatency(median(runP95s))}
          sub={`median of per-run p95 across ${formatCount(runsWithLatency)} runs, not a pooled percentile`}
        />
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Model ranking</h2>
          <span className="spacer" />
          <Link to="/models">quality against cost and latency</Link>
        </div>
        {models.isPending ? (
          <Loading what="the model summary" />
        ) : rows.length === 0 ? (
          <EmptyState title="No models to rank yet">
            The ranking pools every stored run by provider and model. It appears as soon as one
            run has been recorded.
          </EmptyState>
        ) : (
          <>
            <ScrollRegion label="Model ranking table">
              <table className="data">
                <caption className="sr-only">
                  Models ranked by pooled pass rate, with the 95% Wilson interval the API
                  computed, run and case counts, cost per case and median run p95 latency.
                </caption>
                <thead>
                  <tr>
                    <th scope="col" className="num">
                      #
                    </th>
                    <th scope="col">Provider / model</th>
                    <th scope="col">Pass rate (95% Wilson)</th>
                    <th scope="col" className="num">
                      Runs
                    </th>
                    <th scope="col" className="num">
                      Cases
                    </th>
                    <th scope="col" className="num">
                      Cost / case
                    </th>
                    <th scope="col" className="num">
                      Median run P95
                    </th>
                    <th scope="col">Last run</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows]
                    .sort((a, b) => (b.pass_rate ?? -1) - (a.pass_rate ?? -1))
                    .map((row, index) => (
                      <tr key={`${row.provider}/${row.model}`}>
                        <td className="num dim">{index + 1}</td>
                        <td className="mono nowrap">
                          {row.provider} / {row.model}
                        </td>
                        <td>
                          <ConfidenceInterval
                            rate={row.pass_rate}
                            interval={row.pass_rate_ci}
                            n={row.n_pass_denominator}
                          />
                        </td>
                        <td className="num">{formatCount(row.n_runs)}</td>
                        <td className="num">{formatCount(row.n_cases)}</td>
                        <td className="num">
                          {row.cost_per_case === null || row.cost_per_case === undefined ? (
                            <span className="dim">unknown</span>
                          ) : (
                            formatCost(row.cost_per_case)
                          )}
                        </td>
                        <td className="num">{formatLatency(row.median_run_p95_ms)}</td>
                        <td className="mono nowrap dim">{formatTimestamp(row.last_run_at)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </ScrollRegion>
            <p className="panel-note">
              Rates are pooled over every stored run of that model, not averaged across runs,
              and the interval is the Wilson interval the summary endpoint returns over that
              pooled denominator. Cost figures exclude cases that carried no price rather than
              counting them as free.
            </p>
          </>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Regressions</h2>
          <span className="dim">runs that declared a baseline</span>
        </div>
        {withBaseline.length === 0 ? (
          <EmptyState title="No recent run declares a baseline">
            A run carries a baseline when it is created with one, and the gate result then
            appears here. Any two runs can still be compared directly from the Comparison view.
          </EmptyState>
        ) : (
          <>
            <ScrollRegion label="Regression strip">
              <table className="data">
                <caption className="sr-only">
                  Runs that declared a baseline, with the gate verdict and the pass-rate delta
                  with its interval.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Candidate run</th>
                    <th scope="col">Baseline run</th>
                    <th scope="col">Verdict</th>
                    <th scope="col">Pass-rate delta (95% interval)</th>
                    <th scope="col" className="num">
                      Newly failing
                    </th>
                    <th scope="col" />
                  </tr>
                </thead>
                <tbody>
                  {withBaseline.map((detail, index) => {
                    const report: RegressionReport | undefined = comparisons[index]?.data;
                    const passCheck = report?.checks.find((c) => c.metric === "pass_rate");
                    const baselineId = detail.run.baseline_run_id ?? "";
                    return (
                      <tr key={detail.run.id}>
                        <td className="mono">
                          <Link to={`/runs/${detail.run.id}`}>
                            {detail.run.label ?? shortHash(detail.run.id)}
                          </Link>
                        </td>
                        <td className="mono">
                          <Link to={`/runs/${baselineId}`}>{shortHash(baselineId)}</Link>
                        </td>
                        <td>
                          {report === undefined ? (
                            <span className="dim">loading</span>
                          ) : report.verdict === "pass" ? (
                            <Badge tone="pass">pass</Badge>
                          ) : report.verdict === "fail" ? (
                            <Badge tone="fail">fail</Badge>
                          ) : report.verdict === "warn" ? (
                            <Badge tone="warn">warn</Badge>
                          ) : (
                            <Badge tone="muted">incomparable</Badge>
                          )}
                        </td>
                        <td>
                          {report === undefined ? (
                            <span className="dim">loading</span>
                          ) : (
                            <PassRateDelta
                              check={passCheck}
                              intervalsOverlapFlag={report.intervals_overlap}
                            />
                          )}
                        </td>
                        <td className="num">
                          {formatCount(report?.summary.newly_failing.length ?? null)}
                        </td>
                        <td>
                          <Link
                            to={`/compare?baseline=${encodeURIComponent(baselineId)}&candidate=${encodeURIComponent(detail.run.id)}`}
                          >
                            full comparison
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </ScrollRegion>
            <p className="panel-note">
              The verdict is the backend&apos;s deterministic point-estimate answer. The
              interval beside the delta is advisory and never changes it. Baseline and
              candidate rates are on the full comparison, where each carries its own interval.
            </p>
          </>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Recent runs</h2>
          <span className="spacer" />
          <Link to="/runs">all runs</Link>
        </div>
        {runs.isPending ? (
          <Loading what="runs" />
        ) : recent.length === 0 ? (
          <NoRunsYet />
        ) : (
          <RunTable runs={recent} label="Recent runs table" />
        )}
      </div>
    </>
  );
}
