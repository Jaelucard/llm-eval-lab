/**
 * Latency percentiles as bars, with the table they summarise directly beneath.
 *
 * The chart adds one thing the table cannot: the shape of the tail, seen at a
 * glance. It does not replace the numbers, and it does not smooth them into a
 * curve — a percentile is a point, and drawing a line between p50 and p99 would
 * invent values that were never measured.
 *
 * Which percentiles are under-powered is read from the backend's own
 * `low_confidence` list and never re-derived here; the sample floors in
 * `lib/gates.ts` exist only so the badge can say what the floor was. A
 * low-confidence bar is marked twice over, by colour and by a dashed outline,
 * so the chart does not depend on colour alone.
 */

import { Badge } from "./Badge";
import { EmptyState } from "./EmptyState";
import { ScrollRegion } from "./ScrollRegion";
import type { LatencyStats } from "../api/client";
import { PERCENTILE_MIN_SAMPLES, isGatedPercentile } from "../lib/gates";
import { formatCount, formatLatency } from "../lib/format";

/** Every row the chart draws, gated or not. `max_ms` is never gated. */
type LatencyRowKey = "p50_ms" | "p90_ms" | "p95_ms" | "p99_ms" | "max_ms";

interface Row {
  key: LatencyRowKey;
  label: string;
  value: number | null;
}

const WIDTH = 640;
const ROW_HEIGHT = 26;
const LEFT = 54;
const RIGHT = 76;
const TOP = 12;

export interface LatencyChartProps {
  latency: LatencyStats;
}

/** What the badge's tooltip says, given which percentile is flagged. */
function gateExplanation(key: LatencyRowKey, n: number): string {
  if (!isGatedPercentile(key)) {
    return `The backend flagged this statistic as low confidence at n=${String(n)}.`;
  }
  return (
    `This percentile needs at least ${String(PERCENTILE_MIN_SAMPLES[key])} samples ` +
    `to be stable; this run has ${String(n)}.`
  );
}

export function LatencyChart({ latency }: LatencyChartProps) {
  const rows: Row[] = [
    { key: "p50_ms", label: "p50", value: latency.p50_ms },
    { key: "p90_ms", label: "p90", value: latency.p90_ms },
    { key: "p95_ms", label: "p95", value: latency.p95_ms },
    { key: "p99_ms", label: "p99", value: latency.p99_ms },
    { key: "max_ms", label: "max", value: latency.max_ms },
  ];

  const lowConfidence = new Set<string>(latency.low_confidence);
  const values = rows.map((r) => r.value).filter((v): v is number => v !== null);

  if (latency.n === 0 || values.length === 0) {
    return (
      <EmptyState title="No latency samples">
        This run recorded no completed provider calls, so there are no latencies to summarise.
        Percentiles appear once at least one case returns.
      </EmptyState>
    );
  }

  const max = Math.max(...values);
  const height = TOP * 2 + rows.length * ROW_HEIGHT;
  const plotWidth = WIDTH - LEFT - RIGHT;
  const scale = (v: number): number => (max === 0 ? 0 : (v / max) * plotWidth);

  return (
    <>
      <ScrollRegion label="Latency percentile chart" variant="chart">
        <svg
          className="chart-svg"
          viewBox={`0 0 ${String(WIDTH)} ${String(height)}`}
          width={WIDTH}
          height={height}
          role="img"
          aria-label="Latency percentiles, with under-powered percentiles outlined and coloured amber"
        >
          <line className="axis-line" x1={LEFT} y1={TOP} x2={LEFT} y2={height - TOP} />
          {rows.map((row, index) => {
            const y = TOP + index * ROW_HEIGHT;
            const low = lowConfidence.has(row.key);
            const width = row.value === null ? 0 : scale(row.value);
            return (
              <g key={row.label}>
                <text className="axis-text" x={LEFT - 8} y={y + 14} textAnchor="end">
                  {row.label}
                </text>
                {row.value === null ? (
                  <text className="axis-text" x={LEFT + 6} y={y + 14}>
                    not computed
                  </text>
                ) : (
                  <>
                    <rect
                      className={low ? "bar-low" : "bar-fill"}
                      x={LEFT + 1}
                      y={y + 4}
                      width={Math.max(width, 1)}
                      height={ROW_HEIGHT - 12}
                      rx={2}
                    />
                    <text className="point-label" x={LEFT + 6 + Math.max(width, 1)} y={y + 14}>
                      {formatLatency(row.value)}
                      {low ? " (low n)" : ""}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </ScrollRegion>

      <ScrollRegion label="Latency statistics table">
        <table className="data">
          <caption className="sr-only">
            Latency mean and percentiles with their sample counts and whether each cleared its
            sample floor.
          </caption>
          <thead>
            <tr>
              <th scope="col">Statistic</th>
              <th scope="col" className="num">
                Value
              </th>
              <th scope="col" className="num">
                Samples
              </th>
              <th scope="col">Confidence</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>mean</td>
              <td className="num">{formatLatency(latency.mean_ms)}</td>
              <td className="num">{formatCount(latency.n)}</td>
              <td className="dim">reported for completeness; the tail is what matters</td>
            </tr>
            {rows.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td className="num">{formatLatency(row.value)}</td>
                <td className="num">{formatCount(latency.n)}</td>
                <td>
                  {lowConfidence.has(row.key) ? (
                    <Badge tone="warn" title={gateExplanation(row.key, latency.n)}>
                      insufficient samples
                    </Badge>
                  ) : (
                    <span className="dim">above its sample floor</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
    </>
  );
}
