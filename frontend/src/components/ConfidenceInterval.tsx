/**
 * A rate, its interval, and its sample count, rendered as one unit.
 *
 * This component exists so that the project's central presentation rule is
 * enforced by construction rather than by review: every pass rate on screen
 * goes through here, and here a point estimate is always accompanied by the
 * interval the backend computed for it and the `n` it was computed from. When
 * the API supplies no interval, that absence is printed — and, where the caller
 * can point at a page that does have one, printed as a link there. An interval
 * is never synthesised in the browser: the runs listing carries no pass-rate
 * denominator, so an interval computed from what it does carry would be over a
 * denominator that is not the rate's own, and narrower than the truth.
 *
 * When `n` is below the gate for the statistic, an "insufficient samples" badge
 * is shown beside the value. That is deliberately different from hiding the
 * number: the reader still sees the estimate and is told not to lean on it.
 */

import type { ReactNode } from "react";

import { Badge, Caveat } from "./Badge";
import { PASS_RATE_MIN_SAMPLES } from "../lib/gates";
import {
  EMPTY,
  formatInterval,
  formatPercent,
  intervalsOverlap,
  type Interval,
} from "../lib/format";

export interface ConfidenceIntervalProps {
  /** The point estimate as a proportion in 0..1. */
  rate: number | null | undefined;
  /**
   * The 95% interval the API returned for that estimate, or `null` when it
   * returned none. Never a value computed in the browser.
   */
  interval: Interval | null | undefined;
  /**
   * The denominator the estimate came from. Required rather than optional: a
   * rate with no visible population is the thing this component prevents.
   * `null` renders an explicit "n unknown" marker.
   */
  n: number | null;
  /**
   * A word qualifying what `n` counts, when it is not the rate's own
   * denominator. The runs listing passes "completed" because that is the only
   * count it carries, and saying so is what keeps the badge honest.
   */
  nLabel?: string;
  /**
   * Shown in place of "(no interval)" when the API supplied none. Used to link
   * to the page that does compute one.
   */
  intervalFallback?: ReactNode;
  /** Minimum `n` this statistic needs before it should be leaned on. */
  minSamples?: number;
  /** A second interval to test for overlap, e.g. the baseline's. */
  compareWith?: Interval | null | undefined;
  /** What the overlap caveat should call the other side. */
  compareLabel?: string;
  digits?: number;
}

export function ConfidenceInterval({
  rate,
  interval,
  n,
  nLabel,
  intervalFallback,
  minSamples = PASS_RATE_MIN_SAMPLES,
  compareWith,
  compareLabel = "baseline",
  digits = 1,
}: ConfidenceIntervalProps) {
  const hasRate = rate !== null && rate !== undefined && Number.isFinite(rate);
  const ci = formatInterval(interval, digits);
  const known = n !== null;
  const underPowered = known && n < minSamples;
  const overlap = intervalsOverlap(interval, compareWith);

  return (
    <span className="chip-row">
      <span className="value">{hasRate ? formatPercent(rate, digits) : EMPTY}</span>
      {ci === null ? (
        <span
          className="ci-missing"
          title="The API returned no interval for this value. None is computed here: an interval over a denominator that is not this rate's own would be narrower than the truth."
        >
          {intervalFallback ?? "(no interval)"}
        </span>
      ) : (
        <span className="ci">{ci}</span>
      )}
      {known ? (
        <span
          className="n-badge"
          title={
            nLabel === undefined
              ? "The population this rate was computed over."
              : `The ${nLabel}-case count this listing carries. It is an upper bound on the rate's own denominator, which only the run's page reports.`
          }
        >
          n={n}
          {nLabel === undefined ? "" : ` ${nLabel}`}
        </span>
      ) : (
        <span className="n-badge" title="This response carried no sample count for the rate.">
          n unknown
        </span>
      )}
      {underPowered ? (
        <Badge
          tone="warn"
          title={`n=${String(n)} is below the ${String(minSamples)}-sample floor for this statistic. A benchmark this size cannot separate small differences from noise.`}
        >
          insufficient samples
        </Badge>
      ) : null}
      {overlap === true ? (
        <Caveat
          title={`This interval overlaps the ${compareLabel} interval, so the difference is not resolved at this sample size.`}
        >
          inconclusive: intervals overlap
        </Caveat>
      ) : null}
    </span>
  );
}
