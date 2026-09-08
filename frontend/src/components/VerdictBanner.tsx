/**
 * The regression gate's answer, stated once and stated plainly.
 *
 * The word is the deterministic point-estimate verdict from the backend: it
 * never changes because of an interval. The overlap caveat sits beside it,
 * because "the gate failed" and "the data can resolve this difference" are two
 * separate claims and conflating them is how a CI gate loses its credibility.
 */

import { Caveat } from "./Badge";
import type { RegressionReport, Verdict } from "../api/client";

const WORDS: Record<Verdict, string> = {
  pass: "PASS",
  warn: "WARN",
  fail: "FAIL",
  incomparable: "INCOMPARABLE",
};

const TONE: Record<Verdict, string> = {
  pass: "verdict-pass",
  warn: "verdict-warn",
  fail: "verdict-fail",
  incomparable: "verdict-muted",
};

export interface VerdictBannerProps {
  report: RegressionReport;
}

export function VerdictBanner({ report }: VerdictBannerProps) {
  const failed = report.checks.filter((c) => c.status === "failed").length;
  const warned = report.checks.filter((c) => c.status === "warning").length;
  const insufficient = report.checks.filter((c) => c.status === "insufficient_data").length;

  return (
    <div className={`verdict ${TONE[report.verdict]}`} role="status" aria-label="Regression verdict">
      <span className="verdict-word">{WORDS[report.verdict]}</span>
      <span className="verdict-detail">
        {report.comparable ? (
          <>
            {report.checks.length} checks evaluated on point estimates:{" "}
            {failed} failed, {warned} warning, {insufficient} without enough data.
            {" "}Comparison mode: {report.mode}, {report.paired_case_count} paired cases.
          </>
        ) : (
          <>These runs cannot be compared. {report.incomparable_reason ?? "No reason given."}</>
        )}
      </span>
      {report.intervals_overlap === true ? (
        <Caveat title="The overall baseline and candidate pass-rate intervals overlap. The gate verdict still stands: it is computed from point estimates by design.">
          inconclusive: intervals overlap
        </Caveat>
      ) : null}
      {!report.suite_hash_match ? (
        <Caveat title="The two runs were executed against different suite content. Cases that changed are listed below.">
          suite hash differs
        </Caveat>
      ) : null}
    </div>
  );
}
