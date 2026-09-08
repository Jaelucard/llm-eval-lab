/**
 * TanStack Query hooks. All data fetching in the app goes through here.
 *
 * Two behaviours are worth naming. Run status polls every two seconds while a
 * run is `pending` or `running` and stops dead on any terminal status, which is
 * the whole of the live-progress story: no websocket, no stream. And the
 * metrics query absorbs the `X-Metrics-Materialized: false` window itself, by
 * refetching once after a short pause, so a run that has just finished never
 * flashes a scary "metrics unavailable" state at the reader on its way to the
 * real numbers.
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";

import {
  api,
  isTerminal,
  type AggregateMetrics,
  type CaseListParams,
  type CaseResult,
  type HealthResponse,
  type MetricsResult,
  type ModelSummaryRow,
  type PageOfCases,
  type PageOfRuns,
  type RegressionReport,
  type RunDetail,
  type RunListParams,
  type RunStatusResponse,
  type VersionResponse,
} from "./client";

/** How often a live run's status is re-read, in milliseconds. */
export const STATUS_POLL_MS = 2000;

/** How long to wait before the single re-read of not-yet-materialized metrics. */
export const MATERIALIZE_RETRY_MS = 750;

const STABLE_STALE_MS = 15_000;

export const queryKeys = {
  health: ["health"] as const,
  version: ["version"] as const,
  runs: (params: RunListParams) => ["runs", params] as const,
  run: (id: string) => ["run", id] as const,
  runStatus: (id: string) => ["run", id, "status"] as const,
  runMetrics: (id: string) => ["run", id, "metrics"] as const,
  runCases: (id: string, params: CaseListParams) => ["run", id, "cases", params] as const,
  case: (id: string, caseId: string) => ["run", id, "case", caseId] as const,
  modelSummary: ["models", "summary"] as const,
  compare: (baseline: string, candidate: string) =>
    ["compare", baseline, candidate] as const,
};

export function useHealth(): UseQueryResult<HealthResponse> {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => api.health(signal),
    staleTime: 30_000,
  });
}

export function useVersion(): UseQueryResult<VersionResponse> {
  return useQuery({
    queryKey: queryKeys.version,
    queryFn: ({ signal }) => api.version(signal),
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useRuns(params: RunListParams): UseQueryResult<PageOfRuns> {
  return useQuery({
    queryKey: queryKeys.runs(params),
    queryFn: ({ signal }) => api.listRuns(params, signal),
    staleTime: STABLE_STALE_MS,
  });
}

export function useRun(id: string | undefined): UseQueryResult<RunDetail> {
  return useQuery({
    queryKey: queryKeys.run(id ?? ""),
    queryFn: ({ signal }) => api.getRun(id ?? "", signal),
    enabled: Boolean(id),
    staleTime: STABLE_STALE_MS,
  });
}

/**
 * Poll a run's status while it is live.
 *
 * `refetchInterval` is a function so the polling decision is re-made from the
 * newest response: the moment a terminal status arrives it returns `false` and
 * the timer is dropped.
 */
export function useRunStatus(
  id: string | undefined,
  options: { enabled?: boolean } = {},
): UseQueryResult<RunStatusResponse> {
  const enabled = (options.enabled ?? true) && Boolean(id);
  return useQuery({
    queryKey: queryKeys.runStatus(id ?? ""),
    queryFn: ({ signal }) => api.getRunStatus(id ?? "", signal),
    enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === undefined) return STATUS_POLL_MS;
      return isTerminal(status) ? false : STATUS_POLL_MS;
    },
    refetchIntervalInBackground: false,
  });
}

/**
 * Read a run's aggregate metrics, tolerating the brief window in which the
 * backend reports every case complete but has not yet materialized aggregates.
 */
export function useRunMetrics(
  id: string | undefined,
  options: { enabled?: boolean } = {},
): UseQueryResult<MetricsResult> {
  const enabled = (options.enabled ?? true) && Boolean(id);
  return useQuery({
    queryKey: queryKeys.runMetrics(id ?? ""),
    queryFn: async ({ signal }): Promise<MetricsResult> => {
      const first = await api.getRunMetrics(id ?? "", signal);
      if (first.materialized) return first;
      await new Promise((resolve) => setTimeout(resolve, MATERIALIZE_RETRY_MS));
      return api.getRunMetrics(id ?? "", signal);
    },
    enabled,
    staleTime: STABLE_STALE_MS,
  });
}

export function useRunCases(
  id: string | undefined,
  params: CaseListParams,
  options: { enabled?: boolean } = {},
): UseQueryResult<PageOfCases> {
  return useQuery({
    queryKey: queryKeys.runCases(id ?? "", params),
    queryFn: ({ signal }) => api.listCases(id ?? "", params, signal),
    enabled: (options.enabled ?? true) && Boolean(id),
    staleTime: STABLE_STALE_MS,
  });
}

export function useCase(
  id: string | undefined,
  caseId: string | null,
): UseQueryResult<CaseResult> {
  return useQuery({
    queryKey: queryKeys.case(id ?? "", caseId ?? ""),
    queryFn: ({ signal }) => api.getCase(id ?? "", caseId ?? "", signal),
    enabled: Boolean(id) && Boolean(caseId),
    staleTime: STABLE_STALE_MS,
  });
}

export function useModelSummary(): UseQueryResult<ModelSummaryRow[]> {
  return useQuery({
    queryKey: queryKeys.modelSummary,
    queryFn: ({ signal }) => api.modelSummary(signal),
    staleTime: STABLE_STALE_MS,
  });
}

export function useComparison(
  baseline: string | null,
  candidate: string | null,
): UseQueryResult<RegressionReport> {
  return useQuery({
    queryKey: queryKeys.compare(baseline ?? "", candidate ?? ""),
    queryFn: ({ signal }) => api.compare(baseline ?? "", candidate ?? "", signal),
    enabled: Boolean(baseline) && Boolean(candidate),
    staleTime: STABLE_STALE_MS,
    retry: false,
  });
}

/** Convenience re-export so views do not import the raw metrics type twice. */
export type { AggregateMetrics };
