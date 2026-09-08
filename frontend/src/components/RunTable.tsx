/**
 * The runs listing, used by both the Overview and the Runs view.
 *
 * The pass-rate column carries its sample count and says what that count is,
 * but it carries no interval, and that is deliberate. `RunSummary` reports
 * `pass_rate` and `n_completed` and nothing else: the rate's own denominator
 * excludes score-only cases and, under the default error policy, errored ones,
 * and the backend records `n_pass_denominator` separately precisely because it
 * is not derivable from the counts this listing carries. An interval computed
 * over `n_completed` would therefore be over the wrong population and narrower
 * than the truth, which is the exact over-confidence this dashboard exists to
 * prevent. So the cell says the count is the completed-case count, and links to
 * the run's own page, where the API does supply both the denominator and the
 * interval.
 */

import { Link } from "react-router-dom";

import { Badge, type BadgeTone } from "./Badge";
import { ConfidenceInterval } from "./ConfidenceInterval";
import { NoRunsYet } from "./EmptyState";
import { ScrollRegion } from "./ScrollRegion";
import type { RunStatus, RunSummary } from "../api/client";
import { formatCost, formatCount, formatLatency, formatTimestamp, shortHash } from "../lib/format";

const STATUS_TONE: Record<RunStatus, BadgeTone> = {
  pending: "muted",
  running: "accent",
  completed: "pass",
  partial: "warn",
  failed: "fail",
  cancelled: "muted",
  interrupted: "warn",
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return <Badge tone={STATUS_TONE[status]}>{status}</Badge>;
}

export interface RunTableProps {
  runs: readonly RunSummary[];
  /** Accessible name for the scroll region, so two tables on a page differ. */
  label?: string;
}

export function RunTable({ runs, label = "Runs table" }: RunTableProps) {
  if (runs.length === 0) return <NoRunsYet />;

  return (
    <>
      <ScrollRegion label={label}>
        <table className="data">
          <caption className="sr-only">
            Stored runs with status, model, suite, case counts, pass rate, latency and cost.
          </caption>
          <thead>
            <tr>
              <th scope="col">Status</th>
              <th scope="col">Run</th>
              <th scope="col">Provider / model</th>
              <th scope="col">Suite</th>
              <th scope="col" className="num">
                Cases
              </th>
              <th scope="col" className="num">
                Errors
              </th>
              <th scope="col">Pass rate</th>
              <th scope="col" className="num">
                P95
              </th>
              <th scope="col" className="num">
                Cost
              </th>
              <th scope="col">Created</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td>
                  <RunStatusBadge status={run.status} />
                </td>
                <td>
                  <Link to={`/runs/${run.id}`} className="mono">
                    {run.label ?? shortHash(run.id, 12)}
                  </Link>
                  {run.label === null ? null : (
                    <div className="mono dim">{shortHash(run.id, 12)}</div>
                  )}
                </td>
                <td className="mono nowrap">
                  {run.provider} / {run.model}
                </td>
                <td className="mono">
                  {run.suite_name} <span className="dim">v{run.suite_version}</span>
                  <div className="dim">{shortHash(run.suite_hash, 12)}</div>
                </td>
                <td className="num">
                  {formatCount(run.n_completed)}
                  <span className="dim"> / {formatCount(run.n_cases)}</span>
                </td>
                <td className="num">{formatCount(run.n_errors)}</td>
                <td>
                  <ConfidenceInterval
                    rate={run.pass_rate}
                    interval={null}
                    n={run.n_completed}
                    nLabel="completed"
                    intervalFallback={
                      <Link to={`/runs/${run.id}`}>interval on run detail</Link>
                    }
                  />
                </td>
                <td className="num">{formatLatency(run.p95_ms)}</td>
                <td className="num">{formatCost(run.total_cost)}</td>
                <td className="mono nowrap dim">{formatTimestamp(run.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
      <p className="panel-note">
        This listing carries no pass-rate denominator, so no interval is shown here and none is
        computed in the browser: the rate excludes score-only and, by default, errored cases,
        while the count beside it is the completed-case count. Open a run to see its own
        denominator and its 95% Wilson interval.
      </p>
    </>
  );
}
