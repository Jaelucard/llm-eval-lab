/**
 * The run detail view, and the case-filter accounting behind its match count.
 *
 * The "Errored" filter covers two API case statuses, error and timeout, so it
 * issues two queries and merges them. A disabled TanStack query keeps returning
 * its cached data, so leaving that filter has to drop the timeout total as well
 * as the timeout rows — otherwise the header keeps counting cases the filter no
 * longer describes.
 */

import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunDetail as RunDetailBody } from "../api/client";
import { RunDetail } from "../routes/RunDetail";
import { aggregateMetrics, caseSummary, health, runDetail, version } from "./fixtures";
import { renderAtRoute, stubFetch } from "./helpers";

const RUN_ID = "run-alpha";

/** Twelve cases in all; two errored, one timed out. */
function serveRunDetail(): void {
  stubFetch({
    "/api/health": () => health,
    "/api/version": () => version,
    "/api/runs": (url) => {
      const parts = url.pathname.split("/").filter(Boolean);
      const tail = parts.slice(3).join("/");

      if (tail === "metrics") return aggregateMetrics;
      if (tail === "status") {
        return {
          run_id: RUN_ID,
          status: "completed",
          total: 12,
          completed: 12,
          passed: 9,
          errors: 2,
          cancelled: 0,
          started_at: "2026-02-01T10:00:00Z",
          updated_at: "2026-02-01T10:02:00Z",
          error: null,
          managed: false,
          last_case_id: null,
        };
      }
      if (tail === "cases") {
        const status = url.searchParams.get("status");
        const passed = url.searchParams.get("passed");
        if (status === "timeout") {
          return {
            items: [caseSummary({ case_id: "case-timeout", status: "timeout", passed: false })],
            total: 1,
            limit: 200,
            offset: 0,
          };
        }
        if (status === "error") {
          return {
            items: [
              caseSummary({ case_id: "case-err-1", status: "error", passed: null }),
              caseSummary({ case_id: "case-err-2", status: "error", passed: null }),
            ],
            total: 2,
            limit: 200,
            offset: 0,
          };
        }
        if (passed === "false") {
          return {
            items: [caseSummary({ case_id: "case-fail-1", passed: false, score: 0 })],
            total: 3,
            limit: 200,
            offset: 0,
          };
        }
        return {
          items: Array.from({ length: 12 }, (_, i) =>
            caseSummary({ case_id: `case-${String(i + 1).padStart(3, "0")}` }),
          ),
          total: 12,
          limit: 200,
          offset: 0,
        };
      }
      return runDetail(RUN_ID);
    },
  });
}

function render() {
  return renderAtRoute("/runs/:runId", <RunDetail />, `/runs/${RUN_ID}`);
}

describe("RunDetail", () => {
  it("renders the suite hash, the generation parameters and the filter group", async () => {
    serveRunDetail();
    render();

    expect(await screen.findByText("suite_hash")).toBeInTheDocument();
    expect(screen.getByText("Generation parameters")).toBeInTheDocument();
    expect(screen.getByText("temperature")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Case filter" })).toBeInTheDocument();
  });

  it("drops the timeout total again when the Errored filter is left", async () => {
    serveRunDetail();
    const user = userEvent.setup();
    render();

    await waitFor(() => {
      expect(screen.getByText("12 shown of 12 matching")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Errored" }));
    await waitFor(() => {
      // Two errored plus one timed out: the filter covers both statuses.
      expect(screen.getByText("3 shown of 3 matching")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "All" }));
    await waitFor(() => {
      expect(screen.getByText("12 shown of 12 matching")).toBeInTheDocument();
    });
    // The cached timeout total must not survive the transition as "13".
    expect(screen.queryByText("12 shown of 13 matching")).not.toBeInTheDocument();
  });

  it("names the p95 sample floor correctly on its insufficient-samples badge", async () => {
    serveRunDetail();
    render();

    const table = await screen.findByRole("region", { name: "Latency statistics table" });
    const p95Row = within(table).getByText("p95").closest("tr");
    expect(p95Row).not.toBeNull();
    const badge = within(p95Row as HTMLElement).getByText("insufficient samples");
    expect(badge).toHaveAttribute(
      "title",
      "This percentile needs at least 20 samples to be stable; this run has 12.",
    );
  });

  it("does not flag p90, whose floor is ten samples and not twenty", async () => {
    serveRunDetail();
    render();

    const table = await screen.findByRole("region", { name: "Latency statistics table" });
    const p90Row = within(table).getByText("p90").closest("tr");
    expect(p90Row).not.toBeNull();
    expect(
      within(p90Row as HTMLElement).queryByText("insufficient samples"),
    ).not.toBeInTheDocument();
    expect(p90Row?.textContent).toContain("above its sample floor");
  });

  it("makes its scroll regions reachable by keyboard", async () => {
    serveRunDetail();
    render();

    const cases = await screen.findByRole("region", { name: "Case results table" });
    expect(cases).toHaveAttribute("tabindex", "0");

    const chart = screen.getByRole("region", { name: "Latency percentile chart" });
    expect(chart).toHaveAttribute("tabindex", "0");
  });
});

describe("RunDetail live-to-terminal refresh", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  /**
   * A stateful stub: the run, status and metrics endpoints each start out
   * answering as if the run were still in progress, then flip to their
   * finished answer once the status poll has reported terminal at least
   * once, and the cases endpoint's total climbs to match.
   */
  function serveRunningThenCompleted() {
    const base = runDetail(RUN_ID);
    const runningDetail: RunDetailBody = {
      ...base,
      n_cases: 6,
      n_completed: 3,
      n_passed: 2,
      n_pass_denominator: 3,
      pass_rate: 0.6667,
      run: { ...base.run, status: "running", completed_at: null },
    };
    const completedDetail: RunDetailBody = {
      ...base,
      n_cases: 6,
      n_completed: 6,
      n_passed: 5,
      n_pass_denominator: 6,
      pass_rate: 0.8333,
      run: { ...base.run, status: "completed" },
    };

    const calls = { status: 0, runDetail: 0, metrics: 0, cases: 0 };

    stubFetch({
      "/api/health": () => health,
      "/api/version": () => version,
      "/api/runs": (url) => {
        const parts = url.pathname.split("/").filter(Boolean);
        const tail = parts.slice(3).join("/");

        if (tail === "status") {
          calls.status += 1;
          // Terminal from the second poll onward.
          const status = calls.status < 2 ? "running" : "completed";
          return {
            run_id: RUN_ID,
            status,
            total: 6,
            completed: status === "completed" ? 6 : 3,
            passed: status === "completed" ? 5 : 2,
            errors: 0,
            cancelled: 0,
            started_at: "2026-02-01T10:00:00Z",
            updated_at: "2026-02-01T10:02:00Z",
            error: null,
            managed: false,
            last_case_id: null,
          };
        }
        if (tail === "metrics") {
          calls.metrics += 1;
          return calls.metrics === 1
            ? aggregateMetrics
            : { ...aggregateMetrics, n_completed: 6, mean_score: 0.95, median_score: 1 };
        }
        if (tail === "cases") {
          calls.cases += 1;
          return {
            items: [caseSummary({ case_id: "case-001" })],
            total: calls.cases === 1 ? 3 : 6,
            limit: 200,
            offset: 0,
          };
        }
        calls.runDetail += 1;
        return calls.runDetail === 1 ? runningDetail : completedDetail;
      },
    });

    return calls;
  }

  it("refetches the run, cases and metrics once the poll reports terminal, and stops polling", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const calls = serveRunningThenCompleted();
    render();

    // Starts on the running snapshot, with the live poll banner showing.
    expect(await screen.findByText("running")).toBeInTheDocument();
    expect(screen.getByText("3 / 6")).toBeInTheDocument();
    expect(screen.getByText(/polling every 2s/)).toBeInTheDocument();

    // Let the 2s status poll fire enough times to see the terminal answer.
    await vi.advanceTimersByTimeAsync(2000);
    await vi.advanceTimersByTimeAsync(2000);

    // The completed status, refreshed counts and refreshed metrics all land
    // without a reload, and the live banner is gone.
    await waitFor(() => {
      expect(screen.getByText("completed")).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.queryByText("3 / 6")).not.toBeInTheDocument();
      expect(screen.getByText("6 / 6")).toBeInTheDocument();
    });
    expect(screen.getByText("0.950")).toBeInTheDocument();
    expect(screen.queryByText(/polling every 2s/)).not.toBeInTheDocument();

    expect(calls.runDetail).toBeGreaterThanOrEqual(2);
    expect(calls.metrics).toBeGreaterThanOrEqual(2);
    expect(calls.cases).toBeGreaterThanOrEqual(2);

    // No further status polling after the terminal detail has landed.
    // Exactly two status reads: the initial one and the poll that reported
    // terminal. The refresh above must not have forced a third, and no further
    // tick may issue one either.
    expect(calls.status).toBe(2);
    await vi.advanceTimersByTimeAsync(6000);
    expect(calls.status).toBe(2);
  });
});
