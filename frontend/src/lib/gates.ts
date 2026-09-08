/**
 * The minimum sample counts below which a statistic is flagged, in one place.
 *
 * These mirror the backend's own gates in
 * `src/llm_eval_lab/reporting/statistics.py` (`P50_MIN_N`, `P90_MIN_N`,
 * `P95_MIN_N`, `P99_MIN_N`). The API is the authority on which percentiles are
 * actually low-confidence for a given run — `LatencyStats.low_confidence` is
 * read, never re-derived — but the UI still has to *say* what the gate is when
 * it explains a badge, and a tooltip that states the wrong threshold in a
 * dashboard whose whole claim is honest numbers is not a cosmetic defect.
 * Keeping the numbers here means there is exactly one place to correct if the
 * backend's gates ever move.
 */

/** The latency percentiles the backend gates on sample count. */
export type GatedPercentile = "p50_ms" | "p90_ms" | "p95_ms" | "p99_ms";

/** Minimum samples each gated percentile needs, matching the backend. */
export const PERCENTILE_MIN_SAMPLES: Record<GatedPercentile, number> = {
  p50_ms: 5,
  p90_ms: 10,
  p95_ms: 20,
  p99_ms: 100,
};

/** True when `key` is one of the percentiles the backend gates. */
export function isGatedPercentile(key: string): key is GatedPercentile {
  return key in PERCENTILE_MIN_SAMPLES;
}

/**
 * The sample floor below which a pass rate is flagged as under-powered.
 *
 * Twenty is the point the project's own methodology note picks out: below it a
 * benchmark cannot separate anything smaller than a 40 percentage point
 * difference from noise.
 */
export const PASS_RATE_MIN_SAMPLES = 20;
