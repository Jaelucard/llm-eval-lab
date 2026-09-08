/**
 * Where a run's failures actually came from.
 *
 * Errors are counted over every attempted case, and they are shown as counts
 * rather than folded into the pass rate: a run that errored on a third of its
 * cases and passed the rest is not an 80%-quality model, and this table is what
 * stops the headline number from implying that it is.
 */

import { EmptyState } from "./EmptyState";
import { ScrollRegion } from "./ScrollRegion";
import { EMPTY, formatCount, formatPercent } from "../lib/format";

export interface ErrorBreakdownProps {
  /** `AggregateMetrics.error_breakdown`: error kind to count. */
  breakdown: Record<string, number>;
  /** Total cases attempted, the denominator for the error rate. */
  nCases: number;
  nErrors: number;
  nTimeouts: number;
  nSkipped: number;
  errorRate: number;
  errorPolicy: string;
}

export function ErrorBreakdown({
  breakdown,
  nCases,
  nErrors,
  nTimeouts,
  nSkipped,
  errorRate,
  errorPolicy,
}: ErrorBreakdownProps) {
  const rows = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);

  if (nErrors === 0 && nTimeouts === 0 && nSkipped === 0 && rows.length === 0) {
    return (
      <EmptyState title="No errors, timeouts or skips">
        Every one of the {formatCount(nCases)} cases in this run reached the evaluator.
      </EmptyState>
    );
  }

  return (
    <>
      <ScrollRegion label="Error breakdown table">
        <table className="data">
          <caption className="sr-only">
            Error kinds with their case counts and share of attempted cases.
          </caption>
          <thead>
            <tr>
              <th scope="col">Error kind</th>
              <th scope="col" className="num">
                Cases
              </th>
              <th scope="col" className="num">
                Share of attempted
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([kind, count]) => (
              <tr key={kind}>
                <td className="mono">{kind}</td>
                <td className="num">{formatCount(count)}</td>
                <td className="num">
                  {nCases > 0 ? formatPercent(count / nCases) : EMPTY}
                </td>
              </tr>
            ))}
            <tr>
              <td className="dim">timeouts (included above by kind)</td>
              <td className="num">{formatCount(nTimeouts)}</td>
              <td className="num dim">
                {nCases > 0 ? formatPercent(nTimeouts / nCases) : EMPTY}
              </td>
            </tr>
            <tr>
              <td className="dim">skipped</td>
              <td className="num">{formatCount(nSkipped)}</td>
              <td className="num dim">
                {nCases > 0 ? formatPercent(nSkipped / nCases) : EMPTY}
              </td>
            </tr>
          </tbody>
        </table>
      </ScrollRegion>
      <p className="panel-note">
        Error rate {formatPercent(errorRate)} over {formatCount(nCases)} attempted cases.
        This run&apos;s error_policy is <span className="mono">{errorPolicy}</span>, which is
        what decides whether an errored case sits in the pass-rate denominator.
      </p>
    </>
  );
}
