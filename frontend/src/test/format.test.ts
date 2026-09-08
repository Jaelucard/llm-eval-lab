/**
 * The formatting rules the rest of the dashboard depends on.
 *
 * The load-bearing case is `formatCost(null)`. A run with no price entry must
 * read "unknown"; rendering it as "$0.00" would tell a reader the run was free,
 * which is a different and false claim.
 */

import { describe, expect, it } from "vitest";

import {
  EMPTY,
  UNKNOWN,
  formatCost,
  formatCostPer,
  formatCount,
  formatDuration,
  formatInterval,
  formatLatency,
  formatPValue,
  formatPercent,
  formatPercentagePoints,
  formatRateWithInterval,
  formatScore,
  formatTimestamp,
  formatTokens,
  intervalsOverlap,
  shortHash,
  sumDecimalStrings,
} from "../lib/format";

describe("formatCost", () => {
  it("renders a missing cost as unknown, never as a zero amount", () => {
    expect(formatCost(null)).toBe(UNKNOWN);
    expect(formatCost(undefined)).toBe(UNKNOWN);
    expect(formatCost("")).toBe(UNKNOWN);
    expect(formatCost(null)).not.toBe("$0.00");
  });

  it("renders a genuine zero as a zero amount", () => {
    expect(formatCost("0")).toBe("$0.00");
    expect(formatCost("0.00")).toBe("$0.00");
  });

  it("keeps small amounts legible instead of rounding them away", () => {
    expect(formatCost("0.0000123")).toBe("$0.000012");
    expect(formatCost("0.0421")).toBe("$0.0421");
    expect(formatCost("12.5")).toBe("$12.50");
  });

  it("groups large amounts and keeps the sign", () => {
    expect(formatCost("1234.5")).toBe("$1,234.50");
    expect(formatCost("-2.5")).toBe("-$2.50");
  });

  it("refuses to invent a number from an unparseable string", () => {
    expect(formatCost("not-a-number")).toBe(UNKNOWN);
  });

  it("suffixes a per-unit cost only when the cost is known", () => {
    expect(formatCostPer("0.25", "case")).toBe("$0.2500/case");
    expect(formatCostPer(null, "case")).toBe(UNKNOWN);
  });
});

describe("formatTokens", () => {
  it("renders unknown usage as unknown rather than zero tokens", () => {
    expect(formatTokens(null)).toBe(UNKNOWN);
    expect(formatTokens(undefined)).toBe(UNKNOWN);
    expect(formatTokens(0)).toBe("0");
  });

  it("groups thousands", () => {
    expect(formatTokens(1234567)).toBe("1,234,567");
  });
});

describe("formatLatency", () => {
  it("dashes a missing latency", () => {
    expect(formatLatency(null)).toBe(EMPTY);
    expect(formatLatency(undefined)).toBe(EMPTY);
  });

  it("uses milliseconds below ten seconds and seconds above", () => {
    expect(formatLatency(412)).toBe("412 ms");
    expect(formatLatency(9.4)).toBe("9.4 ms");
    expect(formatLatency(1234.6)).toBe("1,235 ms");
    expect(formatLatency(12500)).toBe("12.50 s");
  });
});

describe("formatPercent and intervals", () => {
  it("dashes a missing rate rather than printing 0%", () => {
    expect(formatPercent(null)).toBe(EMPTY);
    expect(formatPercent(null)).not.toBe("0.0%");
  });

  it("formats a proportion as a percentage", () => {
    expect(formatPercent(0.84)).toBe("84.0%");
    expect(formatPercent(0.84, 0)).toBe("84%");
  });

  it("formats an interval beside its point estimate", () => {
    expect(formatInterval([0.72, 0.92])).toBe("(72.0-92.0%)");
    expect(formatInterval(null)).toBeNull();
    expect(formatRateWithInterval(0.84, [0.72, 0.92])).toBe("84.0% (72.0-92.0%)");
  });

  it("switches the separator when a bound is negative, so a minus is not a range", () => {
    expect(formatInterval([-0.2137, 0.0184])).toBe("(-21.4 to 1.8%)");
    expect(formatInterval([-0.2, -0.05])).toBe("(-20.0 to -5.0%)");
  });

  it("says so explicitly when there is no interval, rather than hiding it", () => {
    expect(formatRateWithInterval(0.84, null)).toBe("84.0% (no interval)");
  });

  it("signs a percentage-point difference", () => {
    expect(formatPercentagePoints(0.023)).toBe("+2.3 pp");
    expect(formatPercentagePoints(-0.023)).toBe("-2.3 pp");
    expect(formatPercentagePoints(null)).toBe(EMPTY);
  });
});

describe("misc formatters", () => {
  it("formats scores and counts", () => {
    expect(formatScore(0.5)).toBe("0.500");
    expect(formatScore(null)).toBe(EMPTY);
    expect(formatCount(1500)).toBe("1,500");
    expect(formatCount(null)).toBe(EMPTY);
  });

  it("clamps tiny p-values instead of printing false precision", () => {
    expect(formatPValue(0.0004)).toBe("<0.001");
    expect(formatPValue(0.0432)).toBe("0.043");
    expect(formatPValue(null)).toBe(EMPTY);
  });

  it("formats an elapsed duration", () => {
    expect(formatDuration("2026-01-01T00:00:00Z", "2026-01-01T00:00:12Z")).toBe("12.0 s");
    expect(formatDuration("2026-01-01T00:00:00Z", "2026-01-01T00:03:05Z")).toBe("3m 5s");
    expect(formatDuration(null, "2026-01-01T00:00:12Z")).toBe(EMPTY);
  });

  it("shortens a hash without pretending it is the whole thing", () => {
    expect(shortHash("abcdef0123456789", 8)).toBe("abcdef01");
    expect(shortHash(null)).toBe(EMPTY);
  });

  it("detects overlapping intervals", () => {
    expect(intervalsOverlap([0.7, 0.9], [0.85, 0.95])).toBe(true);
    expect(intervalsOverlap([0.7, 0.8], [0.85, 0.95])).toBe(false);
    expect(intervalsOverlap(null, [0.85, 0.95])).toBeNull();
  });
});

describe("formatTimestamp", () => {
  it("renders UTC and says so, so two readers see the same time", () => {
    expect(formatTimestamp("2026-02-01T10:00:00Z")).toBe("2026-02-01 10:00:00 UTC");
    // An offset timestamp is normalised to the same zone, not shown as written.
    expect(formatTimestamp("2026-02-01T12:30:00+02:00")).toBe("2026-02-01 10:30:00 UTC");
  });

  it("dashes a missing timestamp and passes an unparseable one through", () => {
    expect(formatTimestamp(null)).toBe(EMPTY);
    expect(formatTimestamp("not a date")).toBe("not a date");
  });
});

describe("sumDecimalStrings", () => {
  it("sums in fixed point rather than accumulating float error", () => {
    const summed = sumDecimalStrings(["0.1", "0.2"]);
    expect(summed.total).toBe("0.300000");
    expect(Number(summed.total)).toBe(0.3);
    expect(summed.counted).toBe(2);
    expect(summed.skipped).toBe(0);
  });

  it("counts unpriced entries instead of treating them as zero", () => {
    const summed = sumDecimalStrings(["1.25", null, "0.75", undefined]);
    expect(summed.total).toBe("2.000000");
    expect(summed.counted).toBe(2);
    expect(summed.skipped).toBe(2);
  });

  it("returns no total at all when nothing was priced", () => {
    const summed = sumDecimalStrings([null, null]);
    expect(summed.total).toBeNull();
    expect(summed.skipped).toBe(2);
    expect(formatCost(summed.total)).toBe(UNKNOWN);
  });
});
