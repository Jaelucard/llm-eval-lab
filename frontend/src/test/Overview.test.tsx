/**
 * The Overview view, and the presentation rule it exists to demonstrate.
 *
 * The central assertion is structural rather than cosmetic: every rate the page
 * renders is walked, and each one must carry both an `n` and an interval (or an
 * explicit statement that no interval was computed). A regression that prints a
 * bare percentage anywhere on this page fails here, not in review.
 */

import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Overview } from "../routes/Overview";
import {
  baselinedReport,
  health,
  modelSummary,
  runDetail,
  runsPage,
  version,
} from "./fixtures";
import { renderWithProviders, stubFetch } from "./helpers";

function serveOverview(): void {
  stubFetch({
    "/api/health": () => health,
    "/api/version": () => version,
    "/api/models/summary": () => modelSummary,
    "/api/runs": (url) => {
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts.length === 3) return runDetail(parts[2] ?? "");
      return runsPage;
    },
  });
}

/** The same Overview, but with the candidate run declaring a baseline. */
function serveOverviewWithBaseline(): void {
  stubFetch({
    "/api/health": () => health,
    "/api/version": () => version,
    "/api/models/summary": () => modelSummary,
    "/api/compare": () => baselinedReport,
    "/api/runs": (url) => {
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts.length === 3) {
        const id = parts[2] ?? "";
        return runDetail(id, id === "run-beta" ? "run-alpha" : null);
      }
      return runsPage;
    },
  });
}

/** Anything that reads as a confidence interval, e.g. "(72.9-85.5%)". */
const INTERVAL_TEXT = /\((-?\d+\.\d+-\d+\.\d+|-?\d+\.\d+ to -?\d+\.\d+)%\)/;

describe("Overview", () => {
  it("renders the recent runs table and the model ranking table", async () => {
    serveOverview();
    renderWithProviders(<Overview />);

    expect(await screen.findByText("Model ranking")).toBeInTheDocument();
    expect(await screen.findByText("Recent runs")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getAllByText("nightly alpha").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText(/fake \/ fake-1/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/fake \/ fake-2/).length).toBeGreaterThan(0);
  });

  it("shows every rate with its sample count and its interval", async () => {
    serveOverview();
    const { container } = renderWithProviders(<Overview />);

    await waitFor(() => {
      expect(container.querySelectorAll(".chip-row").length).toBeGreaterThan(0);
    });

    const rateGroups = Array.from(container.querySelectorAll(".chip-row"));
    expect(rateGroups.length).toBeGreaterThanOrEqual(4);

    for (const group of rateGroups) {
      expect(group.querySelector(".n-badge")).not.toBeNull();
      const interval =
        group.querySelector(".ci") ?? group.querySelector(".ci-missing");
      expect(interval).not.toBeNull();
    }
  });

  it("never renders a percentage outside a group that carries its n and interval", async () => {
    serveOverview();
    const { container } = renderWithProviders(<Overview />);

    await waitFor(() => {
      expect(container.querySelectorAll(".chip-row").length).toBeGreaterThan(0);
    });

    const percentValues = Array.from(container.querySelectorAll(".value")).filter(
      (element) => element.textContent.trim().endsWith("%"),
    );
    expect(percentValues.length).toBeGreaterThan(0);
    for (const element of percentValues) {
      expect(element.closest(".chip-row")).not.toBeNull();
    }
  });

  it("flags a model whose pass rate rests on too few cases", async () => {
    serveOverview();
    const { container } = renderWithProviders(<Overview />);

    await waitFor(() => {
      expect(screen.getAllByText("insufficient samples").length).toBeGreaterThan(0);
    });

    const flagged = screen.getAllByText("insufficient samples")[0];
    const group = flagged?.closest(".chip-row");
    expect(group).not.toBeNull();
    expect(within(group as HTMLElement).getByText("n=12")).toBeInTheDocument();
    expect(container.textContent).toContain("75.0%");
  });

  it("reports unpriced models as a count instead of folding them in as zero", async () => {
    serveOverview();
    const { container } = renderWithProviders(<Overview />);

    await waitFor(() => {
      expect(screen.getAllByText(/fake \/ fake-2/).length).toBeGreaterThan(0);
    });
    expect(screen.getByText("Priced spend")).toBeInTheDocument();

    expect(container.textContent).toContain("carried no price and are excluded");
    // The unpriced run reports "unknown", not a dollar amount of any size.
    expect(screen.getAllByText("unknown").length).toBeGreaterThan(0);
    expect(container.textContent).not.toMatch(/\$0\.00(?!\d)/);
  });

  it("explains itself when no run declares a baseline", async () => {
    serveOverview();
    renderWithProviders(<Overview />);

    expect(
      await screen.findByText("No recent run declares a baseline"),
    ).toBeInTheDocument();
  });

  it("shows the documented empty state when no runs are stored", async () => {
    stubFetch({
      "/api/health": () => health,
      "/api/version": () => version,
      "/api/models/summary": () => [],
      "/api/runs": () => ({ items: [], total: 0, limit: 10, offset: 0 }),
    });
    renderWithProviders(<Overview />);

    expect(await screen.findByText("No runs stored yet")).toBeInTheDocument();
    expect(await screen.findByText("No models to rank yet")).toBeInTheDocument();
  });

  it("shows no interval in the runs table, because the listing carries no denominator", async () => {
    serveOverview();
    renderWithProviders(<Overview />);

    const table = await screen.findByRole("region", { name: "Recent runs table" });
    expect(table.textContent).not.toMatch(INTERVAL_TEXT);
    expect(within(table).getAllByText("interval on run detail").length).toBeGreaterThan(0);
  });

  it("labels the runs table sample count as the completed-case count", async () => {
    serveOverview();
    renderWithProviders(<Overview />);

    const table = await screen.findByRole("region", { name: "Recent runs table" });
    expect(within(table).getByText("n=50 completed")).toBeInTheDocument();
    expect(within(table).getByText("n=12 completed")).toBeInTheDocument();
  });

  it("still shows the API's own interval in the ranking table", async () => {
    serveOverview();
    renderWithProviders(<Overview />);

    const table = await screen.findByRole("region", { name: "Model ranking table" });
    expect(table.textContent).toMatch(INTERVAL_TEXT);
    expect(within(table).getByText("(72.9-85.5%)")).toBeInTheDocument();
  });

  it("renders the regression strip delta with the interval on the delta", async () => {
    serveOverviewWithBaseline();
    renderWithProviders(<Overview />);

    const strip = await screen.findByRole("region", { name: "Regression strip" });

    // The delta is a delta, and the interval beside it is the one the backend
    // computed for that delta, not a baseline-to-candidate pair.
    expect(within(strip).getByText("-10.0 pp")).toBeInTheDocument();
    expect(within(strip).getByText("(-21.4 to 1.8%)")).toBeInTheDocument();
    expect(within(strip).getByText("n=50 paired")).toBeInTheDocument();
    expect(strip.textContent).not.toContain("86.0% to 76.0%");
  });

  it("caveats the regression strip when the intervals overlap, without softening the verdict", async () => {
    serveOverviewWithBaseline();
    renderWithProviders(<Overview />);

    const strip = await screen.findByRole("region", { name: "Regression strip" });
    expect(within(strip).getByText("fail")).toBeInTheDocument();
    expect(within(strip).getByText("inconclusive: intervals overlap")).toBeInTheDocument();
  });

  it("keeps interval styling off anything that is not an interval", async () => {
    serveOverviewWithBaseline();
    const { container } = renderWithProviders(<Overview />);

    await screen.findByRole("region", { name: "Regression strip" });
    for (const element of container.querySelectorAll(".ci")) {
      expect(element.textContent.trim()).toMatch(INTERVAL_TEXT);
    }
  });
});
