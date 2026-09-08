/**
 * A run's cases, filterable to all, failed or errored.
 *
 * The filter is applied by the API rather than in the browser, so the counts
 * under the table are the real ones and not "the ones that happened to be on
 * the page". Selecting a row opens the full case beside the table.
 */

import { Badge, type BadgeTone } from "./Badge";
import { EmptyState } from "./EmptyState";
import { ScrollRegion } from "./ScrollRegion";
import type { CaseResultSummary, CaseStatus } from "../api/client";
import {
  EMPTY,
  formatCost,
  formatCount,
  formatLatency,
  formatScore,
  formatTokens,
} from "../lib/format";

/** The three documented filters, in the order they appear as buttons. */
export type CaseFilter = "all" | "failed" | "errored";

const CASE_FILTERS: readonly { id: CaseFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "failed", label: "Failed" },
  { id: "errored", label: "Errored" },
];

const CASE_STATUS_TONE: Record<CaseStatus, BadgeTone> = {
  ok: "muted",
  error: "fail",
  timeout: "fail",
  skipped: "muted",
  cancelled: "muted",
};

export interface CaseTableProps {
  cases: readonly CaseResultSummary[];
  total: number;
  filter: CaseFilter;
  onFilterChange: (filter: CaseFilter) => void;
  query: string;
  onQueryChange: (query: string) => void;
  selectedCaseId: string | null;
  onSelect: (caseId: string) => void;
}

export function CaseTable({
  cases,
  total,
  filter,
  onFilterChange,
  query,
  onQueryChange,
  selectedCaseId,
  onSelect,
}: CaseTableProps) {
  return (
    <>
      <div className="panel-head">
        <h2>Cases</h2>
        <div className="controls" role="group" aria-label="Case filter">
          {CASE_FILTERS.map((option) => (
            <button
              key={option.id}
              type="button"
              className="seg"
              aria-pressed={filter === option.id}
              onClick={() => {
                onFilterChange(option.id);
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
        <input
          type="search"
          aria-label="Search cases"
          placeholder="case id contains"
          value={query}
          onChange={(event) => {
            onQueryChange(event.target.value);
          }}
        />
        <span className="spacer" />
        <span className="dim mono">
          {formatCount(cases.length)} shown of {formatCount(total)} matching
        </span>
      </div>

      {cases.length === 0 ? (
        <EmptyState title="No cases match this filter">
          {filter === "failed"
            ? "No case in this run failed its evaluators under the current search."
            : filter === "errored"
              ? "No case in this run errored or timed out under the current search."
              : "This run has no stored cases yet. A run that is still pending writes its first case when the first provider call returns."}
        </EmptyState>
      ) : (
        <ScrollRegion label="Case results table">
          <table className="data">
            <caption className="sr-only">
              One row per case, with its status, verdict, score, latency, tokens, cost,
              category, tags and attempt count.
            </caption>
            <thead>
              <tr>
                <th scope="col">Case</th>
                <th scope="col">Status</th>
                <th scope="col">Result</th>
                <th scope="col" className="num">
                  Score
                </th>
                <th scope="col" className="num">
                  Latency
                </th>
                <th scope="col" className="num">
                  Tokens
                </th>
                <th scope="col" className="num">
                  Cost
                </th>
                <th scope="col">Category</th>
                <th scope="col">Tags</th>
                <th scope="col" className="num">
                  Attempts
                </th>
              </tr>
            </thead>
            <tbody>
              {cases.map((row) => (
                <tr key={row.case_id}>
                  <td className="mono">
                    <button
                      type="button"
                      className="link"
                      aria-pressed={selectedCaseId === row.case_id}
                      onClick={() => {
                        onSelect(row.case_id);
                      }}
                    >
                      {row.case_id}
                    </button>
                  </td>
                  <td>
                    <Badge tone={CASE_STATUS_TONE[row.status]}>{row.status}</Badge>
                    {row.error_kind === null ? null : (
                      <div className="mono dim">{row.error_kind}</div>
                    )}
                  </td>
                  <td>
                    {row.passed === null ? (
                      <Badge tone="muted" title="This case produced a score but no boolean verdict.">
                        score only
                      </Badge>
                    ) : row.passed ? (
                      <Badge tone="pass">pass</Badge>
                    ) : (
                      <Badge tone="fail">fail</Badge>
                    )}
                  </td>
                  <td className="num">{formatScore(row.score)}</td>
                  <td className="num">{formatLatency(row.latency_ms)}</td>
                  <td className="num">{formatTokens(row.total_tokens)}</td>
                  <td className="num">
                    {row.priced === false ? (
                      <span className="dim" title="No price entry covered this case's model.">
                        unpriced
                      </span>
                    ) : (
                      formatCost(row.total_cost)
                    )}
                  </td>
                  <td className="mono">{row.category ?? EMPTY}</td>
                  <td className="mono dim">{row.tags.join(", ") || EMPTY}</td>
                  <td className="num">{formatCount(row.attempts)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollRegion>
      )}
    </>
  );
}
