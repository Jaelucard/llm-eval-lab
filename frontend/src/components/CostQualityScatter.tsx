/**
 * Quality against a cost-like axis, with a vertical error bar on every point.
 *
 * The error bar is the point of the chart. Two models whose pass-rate intervals
 * overlap are not ranked by this picture, however far apart their dots look,
 * and drawing the interval is the only way to say so without words. Points
 * whose sample count is below the gate are drawn as an amber triangle rather
 * than a filled circle, so the flag survives for a reader who cannot separate
 * the two colours, and every point is repeated in the table beneath.
 *
 * Point labels are dropped once the chart holds more than a handful of models,
 * because at that density they overplot each other and hide the marks they name.
 * The per-point tooltip and the raw table carry the identity either way.
 */

import { EmptyState } from "./EmptyState";
import { ScrollRegion } from "./ScrollRegion";
import { formatPercent } from "../lib/format";
import type { Interval } from "../lib/format";

export interface ScatterPoint {
  id: string;
  label: string;
  /** The cost or latency value on the horizontal axis. `null` drops the point. */
  x: number | null;
  /** The pass rate in 0..1 on the vertical axis. */
  y: number | null;
  /** The pass rate's 95% interval, drawn as the error bar. */
  interval: Interval | null;
  /** Denominator behind the rate, shown in the point's tooltip. */
  n: number;
  /** True when `n` is below the gate for this statistic. */
  lowConfidence: boolean;
}

export interface CostQualityScatterProps {
  points: readonly ScatterPoint[];
  xLabel: string;
  yLabel: string;
  /** How an x value should read on the axis and in the tooltip. */
  formatX: (value: number) => string;
  /** What to say when every point is missing its x value. */
  emptyTitle: string;
  emptyBody: string;
}

/** Above this many plotted points, labels overlap more than they inform. */
const MAX_LABELLED_POINTS = 8;

const WIDTH = 640;
const HEIGHT = 320;
const LEFT = 56;
const RIGHT = 18;
const TOP = 14;
const BOTTOM = 46;

function niceMax(value: number): number {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
}

export function CostQualityScatter({
  points,
  xLabel,
  yLabel,
  formatX,
  emptyTitle,
  emptyBody,
}: CostQualityScatterProps) {
  const usable = points.filter(
    (p): p is ScatterPoint & { x: number; y: number } =>
      p.x !== null && p.y !== null && Number.isFinite(p.x) && Number.isFinite(p.y),
  );

  if (usable.length === 0) {
    return <EmptyState title={emptyTitle}>{emptyBody}</EmptyState>;
  }

  const xMax = niceMax(Math.max(...usable.map((p) => p.x)));
  const plotWidth = WIDTH - LEFT - RIGHT;
  const plotHeight = HEIGHT - TOP - BOTTOM;
  const xPos = (v: number): number => LEFT + (xMax === 0 ? 0 : (v / xMax) * plotWidth);
  const yPos = (v: number): number => TOP + plotHeight - v * plotHeight;

  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax);

  const labelled = usable.length <= MAX_LABELLED_POINTS;

  return (
    <ScrollRegion label={`${yLabel} against ${xLabel} chart`} variant="chart">
      <svg
        className="chart-svg"
        viewBox={`0 0 ${String(WIDTH)} ${String(HEIGHT)}`}
        width={WIDTH}
        height={HEIGHT}
        role="img"
        aria-label={`${yLabel} against ${xLabel}, with 95% intervals`}
      >
        {yTicks.map((tick) => (
          <g key={`y-${String(tick)}`}>
            <line
              className="grid-line"
              x1={LEFT}
              y1={yPos(tick)}
              x2={WIDTH - RIGHT}
              y2={yPos(tick)}
            />
            <text
              className="axis-text"
              x={LEFT - 8}
              y={yPos(tick) + 3}
              textAnchor="end"
            >
              {formatPercent(tick, 0)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <text
            key={`x-${String(tick)}`}
            className="axis-text"
            x={xPos(tick)}
            y={HEIGHT - BOTTOM + 16}
            textAnchor="middle"
          >
            {formatX(tick)}
          </text>
        ))}

        <line
          className="axis-line"
          x1={LEFT}
          y1={TOP}
          x2={LEFT}
          y2={HEIGHT - BOTTOM}
        />
        <line
          className="axis-line"
          x1={LEFT}
          y1={HEIGHT - BOTTOM}
          x2={WIDTH - RIGHT}
          y2={HEIGHT - BOTTOM}
        />

        <text
          className="axis-title"
          x={LEFT + plotWidth / 2}
          y={HEIGHT - 8}
          textAnchor="middle"
        >
          {xLabel}
        </text>
        <text
          className="axis-title"
          x={12}
          y={TOP + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${String(TOP + plotHeight / 2)})`}
        >
          {yLabel}
        </text>

        {usable.map((point) => {
          const cx = xPos(point.x);
          const cy = yPos(point.y);
          const interval = point.interval;
          return (
            <g key={point.id}>
              {interval === null ? null : (
                <>
                  <line
                    className="errorbar"
                    x1={cx}
                    y1={yPos(interval[0])}
                    x2={cx}
                    y2={yPos(interval[1])}
                  />
                  <line
                    className="errorbar"
                    x1={cx - 4}
                    y1={yPos(interval[0])}
                    x2={cx + 4}
                    y2={yPos(interval[0])}
                  />
                  <line
                    className="errorbar"
                    x1={cx - 4}
                    y1={yPos(interval[1])}
                    x2={cx + 4}
                    y2={yPos(interval[1])}
                  />
                </>
              )}
              {point.lowConfidence ? (
                <polygon
                  className="point-low"
                  points={`${String(cx)},${String(cy - 5)} ${String(cx + 5)},${String(cy + 4)} ${String(cx - 5)},${String(cy + 4)}`}
                >
                  <title>
                    {`${point.label}: ${formatPercent(point.y)} pass, n=${String(point.n)} (below the sample floor), ${formatX(point.x)}`}
                  </title>
                </polygon>
              ) : (
                <circle className="point-fill" cx={cx} cy={cy} r={4}>
                  <title>
                    {`${point.label}: ${formatPercent(point.y)} pass, n=${String(point.n)}, ${formatX(point.x)}`}
                  </title>
                </circle>
              )}
              {labelled ? (
                <text className="point-label" x={cx + 8} y={cy - 6}>
                  {point.label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span>Vertical bars are 95% Wilson intervals on the pass rate.</span>
        <span>Amber triangles are below the sample floor; circles are above it.</span>
        {labelled ? null : (
          <span>
            Labels are omitted above {MAX_LABELLED_POINTS} models; hover a mark or read the
            table below.
          </span>
        )}
      </div>
    </ScrollRegion>
  );
}
