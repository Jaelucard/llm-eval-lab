/**
 * The four views rendered against a real API, not a fixture.
 *
 * This file is skipped unless `LLM_EVAL_LIVE_API` names a running server, so
 * the default suite and CI stay hermetic. Point it at a local `llm-eval serve`
 * holding at least two runs and it loads each documented route, asserts the
 * elements the phase's exit criteria name, and fails on any console error React
 * emitted while rendering:
 *
 *   LLM_EVAL_LIVE_API=http://127.0.0.1:8000 npm test -- --run src/test/live-routes.test.tsx
 *
 * It is a component-level check in jsdom rather than a browser check: jsdom
 * does not execute the module bundle, so the alternative would have been a
 * headless browser this project does not otherwise depend on. The React tree,
 * the query layer, the real HTTP responses and the real DOM output are all the
 * production ones.
 */

import { screen, waitFor } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import type { RegressionReport, PageOfRuns } from "../api/client";
import { Comparison } from "../routes/Comparison";
import { ModelComparison } from "../routes/ModelComparison";
import { Overview } from "../routes/Overview";
import { RunDetail } from "../routes/RunDetail";
import { renderAtRoute, renderWithProviders } from "./helpers";

const BASE = process.env.LLM_EVAL_LIVE_API ?? "";
const live = BASE !== "";

/** Route relative API paths at the live server, the way a browser would. */
function useLiveFetch(): void {
  const real = globalThis.fetch.bind(globalThis);
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const raw =
      typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    return real(raw.startsWith("/") ? `${BASE}${raw}` : raw, init);
  });
}

let consoleErrors: string[] = [];

describe.skipIf(!live)("live routes", () => {
  let runIds: string[] = [];

  beforeAll(async () => {
    const response = await fetch(`${BASE}/api/runs?limit=10&order=-created_at`);
    const page = (await response.json()) as PageOfRuns;
    runIds = page.items.map((run) => run.id);
    expect(runIds.length).toBeGreaterThanOrEqual(2);
  });

  beforeAll(() => {
    consoleErrors = [];
    vi.spyOn(console, "error").mockImplementation((...args: unknown[]) => {
      consoleErrors.push(args.map((a) => String(a)).join(" "));
    });
  });

  afterAll(() => {
    vi.restoreAllMocks();
  });

  it("renders the Overview with a run table and a ranking carrying rate, n and interval", async () => {
    useLiveFetch();
    const { container } = renderWithProviders(<Overview />);

    await waitFor(
      () => {
        expect(container.querySelectorAll(".chip-row").length).toBeGreaterThan(0);
      },
      { timeout: 5000 },
    );

    expect(screen.getByText("Model ranking")).toBeInTheDocument();
    expect(screen.getByText("Recent runs")).toBeInTheDocument();
    expect(container.querySelectorAll("table.data").length).toBeGreaterThanOrEqual(2);

    for (const group of container.querySelectorAll(".chip-row")) {
      expect(group.querySelector(".n-badge")).not.toBeNull();
      expect(group.querySelector(".ci") ?? group.querySelector(".ci-missing")).not.toBeNull();
    }
    expect(consoleErrors).toEqual([]);
  });

  it("renders a run's suite hash, generation parameters and filterable case table", async () => {
    useLiveFetch();
    const runId = runIds[0] ?? "";
    const { container } = renderAtRoute("/runs/:runId", <RunDetail />, `/runs/${runId}`);

    await waitFor(
      () => {
        expect(screen.getByText("suite_hash")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );

    expect(screen.getByText("Generation parameters")).toBeInTheDocument();
    expect(screen.getByText("temperature")).toBeInTheDocument();
    expect(screen.getByText("max_output_tokens")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Case filter" })).toBeInTheDocument();

    await waitFor(
      () => {
        expect(container.textContent).toContain("shown of");
      },
      { timeout: 5000 },
    );
    expect(consoleErrors).toEqual([]);
  });

  it("renders the comparison verdict banner and per-check table", async () => {
    useLiveFetch();
    const candidate = runIds[0] ?? "";
    const baseline = runIds[1] ?? "";
    const report = (await (
      await fetch(
        `${BASE}/api/compare?baseline_run_id=${baseline}&candidate_run_id=${candidate}`,
      )
    ).json()) as RegressionReport;

    renderWithProviders(<Comparison />, [
      `/compare?baseline=${baseline}&candidate=${candidate}`,
    ]);

    const banner = await screen.findByRole(
      "status",
      { name: "Regression verdict" },
      { timeout: 5000 },
    );
    expect(banner.textContent).toContain(report.verdict.toUpperCase());
    expect(screen.getByText("Checks")).toBeInTheDocument();
    expect(screen.getAllByText("overall pass rate").length).toBeGreaterThan(0);
    expect(consoleErrors).toEqual([]);
  });

  it("renders both model scatter plots with error bars and the raw table", async () => {
    useLiveFetch();
    const { container } = renderWithProviders(<ModelComparison />, ["/models"]);

    await waitFor(
      () => {
        expect(screen.getByText("Raw figures")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );

    expect(screen.getByText("Quality against cost")).toBeInTheDocument();
    expect(screen.getByText("Quality against latency")).toBeInTheDocument();
    expect(container.querySelectorAll("svg.chart-svg").length).toBe(2);
    expect(container.querySelectorAll("line.errorbar").length).toBeGreaterThan(0);
    expect(consoleErrors).toEqual([]);
  });
});
