/**
 * One run, in full: what was asked, what came back, and how confident to be.
 *
 * The page is ordered the way an engineer reads a failure. Identity first —
 * which model, which suite, which suite hash, which generation parameters —
 * because a number is meaningless until you know what produced it. Then the
 * aggregates with their intervals, then latency, cost and errors, then the case
 * table and the single case under inspection with its evaluator results and, if
 * the case was model-graded, the judge's full provenance.
 *
 * While the run is still live, status polls every two seconds and stops on the
 * first terminal status. That poll response is the first thing to learn a run
 * has finished, well before the 15s staleTime on the run/cases/metrics queries
 * would naturally refetch them, so the moment the poll reports terminal this
 * view invalidates those three so the page shows the finished run's real
 * status, counts and numbers instead of the stale in-progress snapshot.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import { isTerminal, type EvaluationResult } from "../api/client";
import {
  queryKeys,
  useCase,
  useRun,
  useRunCases,
  useRunMetrics,
  useRunStatus,
} from "../api/queries";
import { Badge } from "../components/Badge";
import { CaseTable, type CaseFilter } from "../components/CaseTable";
import { ConfidenceInterval } from "../components/ConfidenceInterval";
import { EmptyState } from "../components/EmptyState";
import { ErrorBreakdown } from "../components/ErrorBreakdown";
import { JudgeVerdictPanel } from "../components/JudgeVerdictPanel";
import { LatencyChart } from "../components/LatencyChart";
import { MethodTag } from "../components/MethodTag";
import { MetricTile } from "../components/MetricTile";
import { OutputBlock } from "../components/OutputBlock";
import { Loading, QueryError } from "../components/QueryState";
import { RunStatusBadge } from "../components/RunTable";
import { ScrollRegion } from "../components/ScrollRegion";
import {
  EMPTY,
  formatCost,
  formatCount,
  formatDuration,
  formatLatency,
  formatNumber,
  formatPercent,
  formatScore,
  formatTimestamp,
  formatTokens,
} from "../lib/format";

const CASE_PAGE_SIZE = 200;

function evaluationTone(status: EvaluationResult["status"]) {
  if (status === "passed") return "pass" as const;
  if (status === "failed") return "fail" as const;
  if (status === "error") return "warn" as const;
  return "muted" as const;
}

export function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [filter, setFilter] = useState<CaseFilter>("all");
  const [query, setQuery] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);

  const queryClient = useQueryClient();
  const run = useRun(runId);
  // Whether the cached run detail still looks live: this is what keeps the
  // status poll enabled, so it must not itself depend on the poll's answer.
  const cachedLive = run.data ? !isTerminal(run.data.run.status) : false;
  const status = useRunStatus(runId, { enabled: cachedLive });
  const polledStatus = status.data?.status;
  const polledTerminal = polledStatus !== undefined && isTerminal(polledStatus);
  // The header stops claiming "polling every 2s" as soon as the poll itself
  // reports terminal, even before the run/cases/metrics refetch below lands.
  const live = cachedLive && !polledTerminal;
  const metrics = useRunMetrics(runId);

  // The status poll is the first signal that a run has finished, well before
  // the run/cases/metrics queries' own staleTime would refetch them. Fire the
  // refresh once per run, the first time the poll reports terminal, so the
  // page never sits on a stale status badge, stale counts or stale metrics.
  //
  // One predicate call, not `invalidateQueries({ queryKey: queryKeys.run(id) })`:
  // TanStack matches keys by PREFIX unless told otherwise, so that shorter key
  // would also match `["run", id, "status"]` and force one more status fetch
  // the poll's own terminal check has already made unnecessary. The predicate
  // names the run's other queries (detail, metrics, cases, case) and leaves the
  // status query alone.
  const invalidatedForRunRef = useRef<string | null>(null);
  useEffect(() => {
    if (runId === undefined) return;
    if (!polledTerminal) return;
    if (invalidatedForRunRef.current === runId) return;
    invalidatedForRunRef.current = runId;
    const statusKey = queryKeys.runStatus(runId);
    void queryClient.invalidateQueries({
      predicate: (query) =>
        query.queryKey[0] === "run" &&
        query.queryKey[1] === runId &&
        query.queryKey[2] !== statusKey[2],
    });
  }, [runId, polledTerminal, queryClient]);

  const baseParams = {
    limit: CASE_PAGE_SIZE,
    q: query === "" ? undefined : query,
  };
  const primary = useRunCases(runId, {
    ...baseParams,
    ...(filter === "failed" ? { passed: false } : {}),
    ...(filter === "errored" ? { status: "error" as const } : {}),
  });
  const timeouts = useRunCases(
    runId,
    { ...baseParams, status: "timeout" as const },
    { enabled: filter === "errored" },
  );

  const selected = useCase(runId, selectedCaseId);

  if (run.isPending) return <Loading what="this run" />;
  if (run.isError) return <QueryError what="this run" error={run.error} />;

  const detail = run.data;
  const config = detail.run.config;
  const params = config.params;
  const aggregate = metrics.data?.metrics;

  // The timeout query is disabled outside the Errored filter, but a disabled
  // query keeps returning its cached data, so both the rows AND the total have
  // to be gated on the filter. Gating only the rows leaves a stale timeout
  // count in "N shown of M matching" after the reader switches back to All.
  const errored = filter === "errored";
  const primaryItems = primary.data?.items ?? [];
  const timeoutItems = errored ? (timeouts.data?.items ?? []) : [];
  const cases = [...primaryItems, ...timeoutItems];
  const matching =
    (primary.data?.total ?? 0) + (errored ? (timeouts.data?.total ?? 0) : 0);

  return (
    <>
      <div className="page-head">
        <h1>{detail.run.label ?? detail.run.id}</h1>
        <RunStatusBadge status={detail.run.status} />
        {detail.run.warnings.map((warning) => (
          <Badge key={warning} tone="warn" title="Advisory flag recorded by the runner.">
            {warning}
          </Badge>
        ))}
        {live ? (
          <span className="dim mono">
            live: {formatCount(status.data?.completed ?? null)} /{" "}
            {formatCount(status.data?.total ?? null)} complete, polling every 2s
          </span>
        ) : null}
      </div>
      <p className="page-sub">
        <span className="mono">{detail.run.id}</span>
      </p>

      {detail.run.error === null || detail.run.error === undefined ? null : (
        <div className="error-box" role="alert">
          <strong>This run recorded a failure.</strong> {detail.run.error}
        </div>
      )}

      <div className="grid grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Model and benchmark</h2>
          </div>
          <div className="panel-body">
            <dl className="deflist">
              <dt>Provider</dt>
              <dd>{config.provider.provider}</dd>
              <dt>Model</dt>
              <dd>{config.provider.model}</dd>
              <dt>Suite</dt>
              <dd>
                {config.suite_name} v{config.suite_version}
              </dd>
              <dt>suite_hash</dt>
              <dd>{config.suite_hash}</dd>
              <dt>Cases selected</dt>
              <dd>{formatCount(detail.n_cases)}</dd>
              <dt>Library version</dt>
              <dd>
                {config.library_version} (python {config.python_version})
              </dd>
              <dt>Price table</dt>
              <dd>
                {config.price_table_id} v{config.price_table_version}
              </dd>
              <dt>Started</dt>
              <dd>{formatTimestamp(detail.run.started_at)}</dd>
              <dt>Elapsed</dt>
              <dd>{formatDuration(detail.run.started_at, detail.run.completed_at)}</dd>
            </dl>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Generation parameters</h2>
            <span className="dim">what the provider was asked for</span>
          </div>
          <div className="panel-body">
            <dl className="deflist">
              <dt>temperature</dt>
              <dd>{params.temperature ?? "provider default"}</dd>
              <dt>top_p</dt>
              <dd>{params.top_p ?? "provider default"}</dd>
              <dt>max_output_tokens</dt>
              <dd>{params.max_output_tokens ?? "provider default"}</dd>
              <dt>seed</dt>
              <dd>{params.seed ?? "unset"}</dd>
              <dt>response_format</dt>
              <dd>{params.response_format}</dd>
              <dt>stop</dt>
              <dd>{params.stop.join(", ") || EMPTY}</dd>
              <dt>concurrency</dt>
              <dd>{formatCount(config.concurrency)}</dd>
              <dt>case_timeout_s</dt>
              <dd>{formatNumber(config.case_timeout_s)}</dd>
              <dt>error_policy</dt>
              <dd>{config.error_policy}</dd>
              <dt>evaluators</dt>
              <dd>{config.evaluators.map((spec) => spec.type).join(", ") || EMPTY}</dd>
            </dl>
          </div>
        </div>
      </div>

      {metrics.isError ? <QueryError what="aggregate metrics" error={metrics.error} /> : null}

      <div className="grid grid-tiles">
        <MetricTile
          label="Pass rate"
          value={
            <ConfidenceInterval
              rate={detail.pass_rate}
              interval={aggregate?.pass_rate_ci}
              n={detail.n_pass_denominator}
            />
          }
          sub={`${formatCount(detail.n_passed)} passed of ${formatCount(detail.n_pass_denominator)} in the denominator`}
        />
        <MetricTile
          label="Completed"
          value={`${formatCount(detail.n_completed)} / ${formatCount(detail.n_cases)}`}
          sub={`${formatCount(detail.n_errors)} errored`}
        />
        <MetricTile
          label="Mean score"
          value={formatScore(aggregate?.mean_score ?? null)}
          sub={
            aggregate === undefined
              ? "computing"
              : `median ${formatScore(aggregate.median_score)}, n scored ${formatCount(aggregate.n_scored)}`
          }
        />
        <MetricTile
          label="Score-only evaluations"
          value={formatCount(aggregate?.n_score_only ?? null)}
          sub="produced a score but no boolean, so excluded from the pass-rate denominator"
        />
        <MetricTile
          label="Total cost"
          value={formatCost(aggregate?.cost.total_cost)}
          sub={
            aggregate === undefined
              ? "computing"
              : aggregate.cost_coverage === null
                ? "no coverage reported"
                : `${formatPercent(aggregate.cost_coverage)} of cases carried a price`
          }
          title="A run with no price entry reports unknown, never $0.00."
        />
        <MetricTile
          label="Tokens"
          value={formatTokens(aggregate?.tokens.total_tokens ?? null)}
          sub={
            aggregate === undefined
              ? "computing"
              : `${formatCount(aggregate.tokens.n_missing_usage)} cases reported no usage and are excluded`
          }
        />
      </div>

      <div className="grid grid-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Latency</h2>
            <span className="dim">percentiles with their sample gates</span>
          </div>
          {aggregate === undefined ? (
            <Loading what="latency statistics" />
          ) : (
            <LatencyChart latency={aggregate.latency} />
          )}
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Errors</h2>
          </div>
          {aggregate === undefined ? (
            <Loading what="the error breakdown" />
          ) : (
            <ErrorBreakdown
              breakdown={aggregate.error_breakdown}
              nCases={aggregate.n_cases}
              nErrors={aggregate.n_errors}
              nTimeouts={aggregate.n_timeouts}
              nSkipped={aggregate.n_skipped}
              errorRate={aggregate.error_rate}
              errorPolicy={aggregate.error_policy}
            />
          )}
        </div>
      </div>

      {aggregate === undefined ? null : (
        <div className="panel">
          <div className="panel-head">
            <h2>Evaluator results</h2>
            <span className="dim">per evaluator, over this run</span>
          </div>
          {Object.keys(aggregate.by_evaluator).length === 0 ? (
            <EmptyState title="No evaluator results recorded">
              This run stored no evaluations. That happens when every case errored before
              reaching an evaluator.
            </EmptyState>
          ) : (
            <ScrollRegion label="Evaluator results table">
              <table className="data">
                <caption className="sr-only">
                  One row per evaluator, with its pass rate, mean and median score, score-only
                  count and error count.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Evaluator</th>
                    <th scope="col">Type</th>
                    <th scope="col">Pass rate</th>
                    <th scope="col" className="num">
                      Mean score
                    </th>
                    <th scope="col" className="num">
                      Median
                    </th>
                    <th scope="col" className="num">
                      Score only
                    </th>
                    <th scope="col" className="num">
                      Errors
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(aggregate.by_evaluator).map(([id, row]) => (
                    <tr key={id}>
                      <td className="mono">{id}</td>
                      <td className="mono">{row.evaluator_type}</td>
                      <td>
                        <ConfidenceInterval
                          rate={row.pass_rate}
                          interval={null}
                          n={row.n_pass_denominator}
                        />
                      </td>
                      <td className="num">{formatScore(row.mean_score)}</td>
                      <td className="num">{formatScore(row.median_score)}</td>
                      <td className="num">{formatCount(row.n_score_only)}</td>
                      <td className="num">{formatCount(row.n_errors)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollRegion>
          )}
          <p className="panel-note">
            Per-evaluator intervals are not computed by the aggregate endpoint, so these
            rates are shown with their `n` and an explicit note that no interval was
            returned rather than with an interval invented here.
          </p>
        </div>
      )}

      {aggregate === undefined ||
      Object.keys(aggregate.by_category).length === 0 ? null : (
        <div className="panel">
          <div className="panel-head">
            <h2>By category</h2>
          </div>
          <ScrollRegion label="Per-category results table">
            <table className="data">
              <caption className="sr-only">
                Pass rate with its interval and mean score, per case category.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Category</th>
                  <th scope="col">Pass rate</th>
                  <th scope="col" className="num">
                    Mean score
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(aggregate.by_category).map(([name, row]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td>
                      <ConfidenceInterval
                        rate={row.pass_rate}
                        interval={row.pass_rate_ci}
                        n={row.n}
                      />
                    </td>
                    <td className="num">{formatScore(row.mean_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollRegion>
        </div>
      )}

      <div className="panel">
        {primary.isError ? (
          <QueryError what="cases" error={primary.error} />
        ) : (
          <CaseTable
            cases={cases}
            total={matching}
            filter={filter}
            onFilterChange={setFilter}
            query={query}
            onQueryChange={setQuery}
            selectedCaseId={selectedCaseId}
            onSelect={setSelectedCaseId}
          />
        )}
        {errored ? (
          <p className="panel-note">
            The errored filter covers both the error and timeout case statuses, which the
            API reports separately.
          </p>
        ) : null}
      </div>

      {selectedCaseId === null ? null : selected.isPending ? (
        <Loading what={`case ${selectedCaseId}`} />
      ) : selected.isError ? (
        <QueryError what={`case ${selectedCaseId}`} error={selected.error} />
      ) : (
        <>
          <div className="panel">
            <div className="panel-head">
              <h2>Case {selected.data.case_id}</h2>
              <Badge tone={selected.data.passed === true ? "pass" : selected.data.passed === false ? "fail" : "muted"}>
                {selected.data.passed === null ? "score only" : selected.data.passed ? "pass" : "fail"}
              </Badge>
              <span className="spacer" />
              <span className="mono dim">case_hash {selected.data.case_hash}</span>
            </div>
            <div className="panel-body">
              <dl className="deflist">
                <dt>Status</dt>
                <dd>{selected.data.status}</dd>
                <dt>Score</dt>
                <dd>{formatScore(selected.data.score)}</dd>
                <dt>Latency</dt>
                <dd>{formatLatency(selected.data.response?.latency_ms ?? null)}</dd>
                <dt>Attempts</dt>
                <dd>{formatCount(selected.data.attempts)}</dd>
                <dt>Finish reason</dt>
                <dd>{selected.data.response?.finish_reason ?? EMPTY}</dd>
                <dt>Cost</dt>
                <dd>{formatCost(selected.data.cost?.total_cost)}</dd>
                <dt>Tokens</dt>
                <dd>
                  {formatTokens(selected.data.response?.usage.input_tokens ?? null)} in /{" "}
                  {formatTokens(selected.data.response?.usage.output_tokens ?? null)} out
                </dd>
              </dl>
            </div>
            {selected.data.error === null || selected.data.error === undefined ? null : (
              <div className="panel-body">
                <OutputBlock
                  label={`Provider error (${selected.data.error.kind})`}
                  text={selected.data.error.message}
                  meta={selected.data.error.retryable ? "retryable" : "not retryable"}
                />
              </div>
            )}
            <div className="panel-body">
              <OutputBlock
                label="Model output"
                text={selected.data.response?.output_text ?? "(no output recorded)"}
                meta={selected.data.response?.truncated === true ? "truncated" : undefined}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Evaluations for this case</h2>
            </div>
            <ScrollRegion label="Evaluations for the selected case">
              <table className="data">
                <caption className="sr-only">
                  Each evaluator that ran on this case, with its status, normalised and raw
                  scores, weight, similarity method and explanation.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Evaluator</th>
                    <th scope="col">Type</th>
                    <th scope="col">Status</th>
                    <th scope="col" className="num">
                      Score
                    </th>
                    <th scope="col" className="num">
                      Raw
                    </th>
                    <th scope="col" className="num">
                      Weight
                    </th>
                    <th scope="col">Method</th>
                    <th scope="col">Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.data.evaluations.map((evaluation) => (
                    <tr key={evaluation.evaluator_id}>
                      <td className="mono">{evaluation.evaluator_id}</td>
                      <td className="mono">{evaluation.evaluator_type}</td>
                      <td>
                        <Badge tone={evaluationTone(evaluation.status)}>
                          {evaluation.status}
                        </Badge>
                      </td>
                      <td className="num">{formatScore(evaluation.score)}</td>
                      <td className="num">{formatNumber(evaluation.raw_score, 3)}</td>
                      <td className="num">{formatNumber(evaluation.weight, 2)}</td>
                      <td>
                        <MethodTag metadata={evaluation.metadata} />
                      </td>
                      <td>{evaluation.explanation ?? EMPTY}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollRegion>
          </div>

          {selected.data.evaluations
            .filter((evaluation) => evaluation.judge)
            .map((evaluation) =>
              evaluation.judge ? (
                <JudgeVerdictPanel
                  key={`judge-${evaluation.evaluator_id}`}
                  provenance={evaluation.judge}
                  evaluatorId={evaluation.evaluator_id}
                />
              ) : null,
            )}
        </>
      )}
    </>
  );
}
