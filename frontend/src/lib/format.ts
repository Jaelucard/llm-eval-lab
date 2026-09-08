/**
 * Display formatting for every number this dashboard shows.
 *
 * Two rules run through all of it. A missing value is never rendered as a
 * plausible-looking zero: an unpriced run costs "unknown", not "$0.00", and an
 * absent rate is a dash, not 0%. And a rate is never formatted on its own —
 * `formatRateWithInterval` exists so a bare percentage is awkward to produce by
 * accident.
 */

/** What every formatter prints when the backend sent `null`. */
export const UNKNOWN = "unknown";

/** What a numeric cell prints when there is nothing to show at all. */
export const EMPTY = "—";

const NUMBER_LOCALE = "en-US";

/** True when a decimal string from the API parses to a usable finite number. */
function parseDecimal(value: string | null | undefined): number | null {
  if (value === null || value === undefined || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Format a money amount that arrived as a decimal string.
 *
 * `null` means "this run carries no price", which is a different fact from
 * "this run cost nothing" and prints differently. A real zero prints as a zero.
 */
export function formatCost(value: string | null | undefined): string {
  const amount = parseDecimal(value);
  if (amount === null) return UNKNOWN;
  const abs = Math.abs(amount);
  let digits: number;
  if (abs === 0) digits = 2;
  else if (abs < 0.01) digits = 6;
  else if (abs < 1) digits = 4;
  else digits = 2;
  const sign = amount < 0 ? "-" : "";
  return `${sign}$${abs.toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

/** Format a per-unit money amount, keeping small numbers legible. */
export function formatCostPer(value: string | null | undefined, unit: string): string {
  const formatted = formatCost(value);
  return formatted === UNKNOWN ? UNKNOWN : `${formatted}/${unit}`;
}

/** Format a token count. `null` is unknown usage, not zero tokens. */
export function formatTokens(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNKNOWN;
  return Math.round(value).toLocaleString(NUMBER_LOCALE);
}

/** Format an integer count. Unlike tokens, a missing count is a dash. */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return Math.round(value).toLocaleString(NUMBER_LOCALE);
}

/** Format a latency in milliseconds, switching to seconds past 10 s. */
export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return EMPTY;
  if (ms >= 10_000) {
    return `${(ms / 1000).toLocaleString(NUMBER_LOCALE, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })} s`;
  }
  const digits = ms < 10 ? 1 : 0;
  return `${ms.toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} ms`;
}

/**
 * Format a proportion in 0..1 as a percentage.
 *
 * Prefer `formatRateWithInterval` for anything the reader might act on: a bare
 * rate with no `n` beside it is the presentation this project set out to avoid.
 */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return `${(value * 100).toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}%`;
}

/** Format a percentage-point difference, always signed. */
export function formatPercentagePoints(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  const pp = value * 100;
  const sign = pp > 0 ? "+" : "";
  return `${sign}${pp.toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} pp`;
}

/** A confidence interval as it arrives from the API: a `[low, high]` pair. */
export type Interval = readonly [number, number];

/**
 * Format an interval on a 0..1 proportion as `(72.0-92.0%)`.
 *
 * The separator switches to the word "to" when either bound is negative, which
 * happens on every interval around a delta. `(-21.4-1.8%)` is a hyphen sitting
 * next to a minus sign and reads as nonsense; `(-21.4 to 1.8%)` does not. A
 * plain hyphen is kept for the ordinary non-negative case so the common form
 * stays as compact as the methodology note writes it.
 */
export function formatInterval(
  interval: Interval | null | undefined,
  digits = 1,
): string | null {
  if (!interval) return null;
  const [low, high] = interval;
  if (!Number.isFinite(low) || !Number.isFinite(high)) return null;
  const fmt = (v: number): string =>
    (v * 100).toLocaleString(NUMBER_LOCALE, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  const separator = low < 0 || high < 0 ? " to " : "-";
  return `(${fmt(low)}${separator}${fmt(high)}%)`;
}

/**
 * The canonical "point estimate plus its interval" string, e.g.
 * `84.0% (72.0-92.0%)`. When no interval was computed the caller still gets a
 * marker saying so rather than a bare number.
 */
export function formatRateWithInterval(
  rate: number | null | undefined,
  interval: Interval | null | undefined,
  digits = 1,
): string {
  const point = formatPercent(rate, digits);
  const ci = formatInterval(interval, digits);
  return ci === null ? `${point} (no interval)` : `${point} ${ci}`;
}

/** Format a 0..1 score with a fixed width, or a dash when absent. */
export function formatScore(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return value.toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Format a raw number that is not a rate, a score, a cost or a duration. */
export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return value.toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

/**
 * Format a p-value. Anything below 0.001 prints as `<0.001` rather than as a
 * long string of zeroes that invites over-reading.
 */
export function formatPValue(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  if (value < 0.001) return "<0.001";
  return value.toLocaleString(NUMBER_LOCALE, {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  });
}

/**
 * Render an RFC 3339 timestamp in UTC, seconds included, with the zone named.
 *
 * UTC rather than the reader's local zone. Runs are compared across machines
 * and quoted in tickets, and a wall-clock string with no zone marker means two
 * engineers read different times off the same run and neither can tell. The
 * API sends UTC; this shows UTC and says so.
 */
export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return EMPTY;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = (n: number): string => String(n).padStart(2, "0");
  return (
    `${String(date.getUTCFullYear())}-${pad(date.getUTCMonth() + 1)}-` +
    `${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:` +
    `${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())} UTC`
  );
}

/** A compact elapsed-time string between two RFC 3339 timestamps. */
export function formatDuration(
  from: string | null | undefined,
  to: string | null | undefined,
): string {
  if (!from || !to) return EMPTY;
  const start = new Date(from).getTime();
  const end = new Date(to).getTime();
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return EMPTY;
  const seconds = (end - start) / 1000;
  if (seconds < 90) {
    return `${seconds.toLocaleString(NUMBER_LOCALE, {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    })} s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${String(minutes)}m ${String(rest)}s`;
}

/** Shorten a hash for a table cell while keeping it recognisable. */
export function shortHash(value: string | null | undefined, length = 12): string {
  if (!value) return EMPTY;
  return value.length <= length ? value : value.slice(0, length);
}

/** True when two intervals share any point, which makes a delta inconclusive. */
export function intervalsOverlap(
  a: Interval | null | undefined,
  b: Interval | null | undefined,
): boolean | null {
  if (!a || !b) return null;
  return a[0] <= b[1] && b[0] <= a[1];
}

/**
 * Sum money amounts that arrived as decimal strings, staying in fixed point.
 *
 * Adding these as IEEE doubles is how `0.1 + 0.2` becomes `0.30000000000000004`
 * in a cost total. The project keeps money as a Decimal string everywhere else;
 * this scales each value to whole micro-dollars, sums integers, and hands back
 * a string the ordinary cost formatter can render. Values that do not parse are
 * excluded and counted, because a total that quietly absorbs an unparseable
 * amount is worse than one that says how many it left out.
 */
export function sumDecimalStrings(
  values: readonly (string | null | undefined)[],
): { total: string | null; counted: number; skipped: number } {
  const SCALE = 1_000_000;
  let micros = 0;
  let counted = 0;
  let skipped = 0;
  for (const value of values) {
    const parsed = parseDecimal(value);
    if (parsed === null) {
      skipped += 1;
      continue;
    }
    micros += Math.round(parsed * SCALE);
    counted += 1;
  }
  if (counted === 0) return { total: null, counted, skipped };
  return { total: (micros / SCALE).toFixed(6), counted, skipped };
}
