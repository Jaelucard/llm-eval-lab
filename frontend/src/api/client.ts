/**
 * A small typed fetch client over the generated OpenAPI types.
 *
 * Every response type here is an alias into `generated-types.ts`, which is
 * produced from the checked-in `openapi.json` snapshot. Nothing in this file
 * restates a backend model by hand, so a backend field that changes shape
 * becomes a type error at the call site rather than a wrong number on screen.
 */

import type { components } from "./generated-types";

type Schemas = components["schemas"];

export type RunSummary = Schemas["RunSummary"];
export type RunDetail = Schemas["RunDetail"];
export type Run = Schemas["Run"];
export type RunStatus = Schemas["RunStatus"];
export type RunStatusResponse = Schemas["RunStatusResponse"];
export type AggregateMetrics = Schemas["AggregateMetrics"];
export type LatencyStats = Schemas["LatencyStats"];
export type CostBreakdown = Schemas["CostBreakdown"];
export type CategoryMetrics = Schemas["CategoryMetrics"];
export type EvaluatorMetrics = Schemas["EvaluatorMetrics"];
export type CaseResultSummary = Schemas["CaseResultSummary"];
export type CaseResult = Schemas["CaseResult"];
export type CaseStatus = Schemas["CaseStatus"];
export type EvaluationResult = Schemas["EvaluationResult"];
export type JudgeProvenance = Schemas["JudgeProvenance"];
export type JudgeVerdict = Schemas["JudgeVerdict"];
export type ModelSummaryRow = Schemas["ModelSummaryRow"];
export type RegressionReport = Schemas["RegressionReport"];
export type CheckOutcome = Schemas["CheckOutcome"];
export type CheckStatus = Schemas["CheckStatus"];
export type CaseDelta = Schemas["CaseDelta"];
export type Verdict = Schemas["Verdict"];
export type HealthResponse = Schemas["HealthResponse"];
export type VersionResponse = Schemas["VersionResponse"];
export type ProblemDetail = Schemas["ProblemDetail"];
export type PageOfRuns = Schemas["Page_RunSummary_"];
export type PageOfCases = Schemas["Page_CaseResultSummary_"];

/** Run statuses that will never change again, so polling can stop. */
export const TERMINAL_STATUSES: readonly RunStatus[] = [
  "completed",
  "partial",
  "failed",
  "cancelled",
  "interrupted",
];

/** True when a run has reached a status that will not change again. */
export function isTerminal(status: RunStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}

/** The header the metrics endpoint uses to say whether its numbers are final. */
export const MATERIALIZED_HEADER = "X-Metrics-Materialized";

/** An HTTP failure carrying the RFC 9457 problem detail when the body had one. */
export class ApiError extends Error {
  readonly status: number;
  readonly problem: ProblemDetail | null;

  constructor(status: number, message: string, problem: ProblemDetail | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.problem = problem;
  }
}

/**
 * True when `error` is a failed request that a bearer token would have fixed.
 *
 * The dashboard never sends `Authorization` (see `README.md#the-dashboard`),
 * so a 401 from a guarded route always means the same thing: the server has
 * `LLM_EVAL_API_TOKEN` set and this request needs one it will never carry.
 */
export function isAuthRequired(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

/** A parsed JSON body together with the response headers that carried it. */
export interface ResponseWithHeaders<T> {
  data: T;
  headers: Headers;
}

/** A query string value the client knows how to serialise. */
export type QueryValue = string | number | boolean | null | undefined;

function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const qs = search.toString();
  return qs === "" ? path : `${path}?${qs}`;
}

function isProblemDetail(value: unknown): value is ProblemDetail {
  return (
    typeof value === "object" &&
    value !== null &&
    "title" in value &&
    typeof value.title === "string"
  );
}

/**
 * Issue a GET and return the parsed body together with the response headers.
 *
 * The headers come back because `X-Metrics-Materialized` is load-bearing: it is
 * how the metrics endpoint says "these aggregates are still being computed",
 * and the caller has to distinguish that from a real answer.
 */
export async function apiGetWithHeaders<T>(
  path: string,
  query?: Record<string, QueryValue>,
  signal?: AbortSignal,
): Promise<ResponseWithHeaders<T>> {
  const url = buildUrl(path, query);
  const init: RequestInit = { headers: { Accept: "application/json" } };
  if (signal) init.signal = signal;
  const response = await fetch(url, init);

  if (!response.ok) {
    let problem: ProblemDetail | null = null;
    try {
      const body: unknown = await response.json();
      if (isProblemDetail(body)) problem = body;
    } catch {
      problem = null;
    }
    const detail = problem
      ? `${problem.title}: ${problem.detail}`
      : `${String(response.status)} ${response.statusText}`;
    throw new ApiError(response.status, detail, problem);
  }

  const data = (await response.json()) as T;
  return { data, headers: response.headers };
}

/** Issue a GET and return only the parsed body. */
export async function apiGet<T>(
  path: string,
  query?: Record<string, QueryValue>,
  signal?: AbortSignal,
): Promise<T> {
  const { data } = await apiGetWithHeaders<T>(path, query, signal);
  return data;
}

/** Filters accepted by the runs listing. */
export interface RunListParams {
  status?: RunStatus | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  suite?: string | undefined;
  limit?: number | undefined;
  offset?: number | undefined;
  order?: string | undefined;
}

/** Filters accepted by a run's case listing. */
export interface CaseListParams {
  status?: CaseStatus | undefined;
  passed?: boolean | undefined;
  tag?: string | undefined;
  category?: string | undefined;
  q?: string | undefined;
  limit?: number | undefined;
  offset?: number | undefined;
}

/** Aggregate metrics plus whether the backend considers them settled. */
export interface MetricsResult {
  metrics: AggregateMetrics;
  /**
   * `false` while the backend is still computing aggregates for a run that
   * already reports every case complete. It is a transient state, not a fault,
   * and the UI refetches rather than warning about it.
   */
  materialized: boolean;
}

export const api = {
  health: (signal?: AbortSignal) => apiGet<HealthResponse>("/api/health", undefined, signal),

  version: (signal?: AbortSignal) => apiGet<VersionResponse>("/api/version", undefined, signal),

  listRuns: (params: RunListParams, signal?: AbortSignal) =>
    apiGet<PageOfRuns>("/api/runs", { ...params }, signal),

  getRun: (id: string, signal?: AbortSignal) =>
    apiGet<RunDetail>(`/api/runs/${encodeURIComponent(id)}`, undefined, signal),

  getRunStatus: (id: string, signal?: AbortSignal) =>
    apiGet<RunStatusResponse>(
      `/api/runs/${encodeURIComponent(id)}/status`,
      undefined,
      signal,
    ),

  getRunMetrics: async (id: string, signal?: AbortSignal): Promise<MetricsResult> => {
    const { data, headers } = await apiGetWithHeaders<AggregateMetrics>(
      `/api/runs/${encodeURIComponent(id)}/metrics`,
      undefined,
      signal,
    );
    return {
      metrics: data,
      materialized: (headers.get(MATERIALIZED_HEADER) ?? "true").toLowerCase() !== "false",
    };
  },

  listCases: (id: string, params: CaseListParams, signal?: AbortSignal) =>
    apiGet<PageOfCases>(`/api/runs/${encodeURIComponent(id)}/cases`, { ...params }, signal),

  getCase: (id: string, caseId: string, signal?: AbortSignal) =>
    apiGet<CaseResult>(
      `/api/runs/${encodeURIComponent(id)}/cases/${encodeURIComponent(caseId)}`,
      undefined,
      signal,
    ),

  modelSummary: (signal?: AbortSignal) =>
    apiGet<ModelSummaryRow[]>("/api/models/summary", undefined, signal),

  compare: (baselineRunId: string, candidateRunId: string, signal?: AbortSignal) =>
    apiGet<RegressionReport>(
      "/api/compare",
      { baseline_run_id: baselineRunId, candidate_run_id: candidateRunId },
      signal,
    ),
};
