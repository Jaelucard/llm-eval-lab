/**
 * Quality against what quality costs: money on one chart, latency on the other.
 *
 * Both charts draw the pass rate's 95% Wilson interval as a vertical bar,
 * because the whole question this view answers — is the cheaper model good
 * enough — turns on whether those bars overlap. The table underneath carries
 * the same numbers in full, including the models the charts had to drop for
 * want of a price or a latency sample, so nothing disappears by being
 * unplottable.
 */

import { Link } from "react-router-dom";

import type { ModelSummaryRow } from "../api/client";
import { useModelSummary } from "../api/queries";
import { Badge } from "../components/Badge";
import { ConfidenceInterval } from "../components/ConfidenceInterval";
import { CostQualityScatter, type ScatterPoint } from "../components/CostQualityScatter";
import { EmptyState } from "../components/EmptyState";
import { Loading, QueryError } from "../components/QueryState";
import { ScrollRegion } from "../components/ScrollRegion";
import { PASS_RATE_MIN_SAMPLES } from "../lib/gates";
import {
  formatCost,
  formatCount,
  formatLatency,
  formatPercent,
  formatTimestamp,
} from "../lib/format";

function toNumber(value: string | null | undefined): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function pointFor(row: ModelSummaryRow, x: number | null): ScatterPoint {
  return {
    id: `${row.provider}/${row.model}`,
    label: row.model,
    x,
    y: row.pass_rate ?? null,
    interval: row.pass_rate_ci ?? null,
    n: row.n_pass_denominator,
    lowConfidence: row.n_pass_denominator < PASS_RATE_MIN_SAMPLES,
  };
}

export function ModelComparison() {
  const models = useModelSummary();

  if (models.isPending) return <Loading what="the model summary" />;
  if (models.isError) return <QueryError what="the model summary" error={models.error} />;

  const rows = models.data;

  if (rows.length === 0) {
    return (
      <>
        <div className="page-head">
          <h1>Model comparison</h1>
        </div>
        <div className="panel">
          <EmptyState title="No models to compare yet">
            This view pools every stored run by provider and model. Record a run, then come
            back: <Link to="/runs">the runs list</Link> shows what is stored.
          </EmptyState>
        </div>
      </>
    );
  }

  const costPoints = rows.map((row) => pointFor(row, toNumber(row.cost_per_case)));
  const latencyPoints = rows.map((row) => pointFor(row, row.median_run_p95_ms ?? null));
  const unpriced = costPoints.filter((point) => point.x === null).length;
  const withoutLatency = latencyPoints.filter((point) => point.x === null).length;

  return (
    <>
      <div className="page-head">
        <h1>Model comparison</h1>
        <span className="dim mono">{formatCount(rows.length)} models pooled</span>
      </div>
      <p className="page-sub">
        Pass rates are pooled over every stored run of each model. Vertical bars are 95%
        Wilson intervals: where two bars overlap, this data does not rank those models.
      </p>

      <div className="grid grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Quality against cost</h2>
            <span className="spacer" />
            {unpriced === 0 ? null : (
              <Badge tone="muted" title="These models carried no price and cannot be placed on a cost axis.">
                {unpriced} unpriced, not plotted
              </Badge>
            )}
          </div>
          <CostQualityScatter
            points={costPoints}
            xLabel="cost per case (USD)"
            yLabel="pass rate"
            formatX={(value) => formatCost(String(value))}
            emptyTitle="No model carries a price"
            emptyBody="Cost per case is unknown for every model here, so there is no cost axis to draw. The price table did not cover these models; the table below still shows their quality."
          />
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Quality against latency</h2>
            <span className="spacer" />
            {withoutLatency === 0 ? null : (
              <Badge tone="muted" title="These models recorded no latency samples.">
                {withoutLatency} without latency, not plotted
              </Badge>
            )}
          </div>
          <CostQualityScatter
            points={latencyPoints}
            xLabel="median run p95 latency (ms)"
            yLabel="pass rate"
            formatX={(value) => formatLatency(value)}
            emptyTitle="No model recorded a p95 latency"
            emptyBody="Every stored run for these models is missing latency samples, so there is no latency axis to draw. The table below still shows their quality."
          />
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Raw figures</h2>
          <span className="dim">the same numbers the charts are drawn from</span>
        </div>
        <ScrollRegion label="Model comparison raw figures table">
          <table className="data">
            <caption className="sr-only">
              Every pooled model with its pass rate and interval, run and case counts, cost
              figures, price coverage and median run p95 latency.
            </caption>
            <thead>
              <tr>
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
                  Total cost
                </th>
                <th scope="col" className="num">
                  Price coverage
                </th>
                <th scope="col" className="num">
                  Median run P95
                </th>
                <th scope="col" className="num">
                  Runs with latency
                </th>
                <th scope="col">Last run</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.provider}/${row.model}`}>
                  <td className="mono nowrap">
                    {row.provider} / {row.model}
                  </td>
                  <td>
                    <ConfidenceInterval
                      rate={row.pass_rate}
                      interval={row.pass_rate_ci}
                      n={row.n_pass_denominator}
                      minSamples={PASS_RATE_MIN_SAMPLES}
                    />
                  </td>
                  <td className="num">{formatCount(row.n_runs)}</td>
                  <td className="num">{formatCount(row.n_cases)}</td>
                  <td className="num">{formatCost(row.cost_per_case)}</td>
                  <td className="num">{formatCost(row.total_cost)}</td>
                  <td className="num">
                    {row.cost_coverage === null || row.cost_coverage === undefined ? (
                      <span className="dim">unknown</span>
                    ) : (
                      formatPercent(row.cost_coverage)
                    )}
                  </td>
                  <td className="num">{formatLatency(row.median_run_p95_ms)}</td>
                  <td className="num">{formatCount(row.n_runs_with_latency)}</td>
                  <td className="mono nowrap dim">{formatTimestamp(row.last_run_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollRegion>
        <p className="panel-note">
          Median run P95 is the median of per-run p95 latencies, not a pooled percentile over
          cases: the summary endpoint does not load per-case latencies. Cost figures exclude
          cases that carried no price, which is what price coverage below 100% reports.
        </p>
      </div>
    </>
  );
}
