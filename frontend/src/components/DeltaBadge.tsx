/**
 * A signed change, coloured by whether it moved the metric the good way.
 *
 * The colour follows the metric's declared direction rather than the sign, so a
 * latency drop reads as an improvement and a cost rise reads as a regression
 * without the reader having to remember which way round each metric goes.
 */

import { EMPTY, formatNumber, formatPercentagePoints } from "../lib/format";

export type Direction = "higher_is_better" | "lower_is_better";

export interface DeltaBadgeProps {
  delta: number | null | undefined;
  direction: Direction;
  /** `pp` renders a 0..1 difference as percentage points; `raw` prints it as-is. */
  unit?: "pp" | "raw" | "ms";
  digits?: number;
}

export function DeltaBadge({ delta, direction, unit = "pp", digits = 1 }: DeltaBadgeProps) {
  if (delta === null || delta === undefined || !Number.isFinite(delta)) {
    return <span className="value delta-flat">{EMPTY}</span>;
  }

  const improved = direction === "higher_is_better" ? delta > 0 : delta < 0;
  const flat = delta === 0;
  const cls = flat ? "delta-flat" : improved ? "delta-up" : "delta-down";

  let text: string;
  if (unit === "pp") {
    text = formatPercentagePoints(delta, digits);
  } else if (unit === "ms") {
    const sign = delta > 0 ? "+" : "";
    text = `${sign}${formatNumber(delta, 0)} ms`;
  } else {
    const sign = delta > 0 ? "+" : "";
    text = `${sign}${formatNumber(delta, digits)}`;
  }

  return (
    <span
      className={`value ${cls}`}
      title={
        flat
          ? "No change."
          : improved
            ? "Moved in the direction this metric calls an improvement."
            : "Moved in the direction this metric calls a regression."
      }
    >
      {text}
    </span>
  );
}
