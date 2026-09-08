/**
 * The Comparison view against a failing gate.
 *
 * Three things are asserted, and each is a rule the project committed to. The
 * banner states the backend's deterministic verdict verbatim. The newly-failing
 * cases are listed by id, because "three regressions" is not actionable and
 * "case-007, case-019, case-042" is. And a check that never reached its minimum
 * sample count renders a badge in place of a difference, so nobody reads a
 * 30-point swing off five cases as if it meant something.
 */

import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Comparison } from "../routes/Comparison";
import { failingReport, health, version } from "./fixtures";
import { renderWithProviders, stubFetch } from "./helpers";

const ROUTE =
  "/compare?baseline=run-alpha&candidate=run-beta";

function serveComparison(report = failingReport): void {
  stubFetch({
    "/api/health": () => health,
    "/api/version": () => version,
    "/api/compare": () => report,
  });
}

function renderComparison(): ReturnType<typeof renderWithProviders> {
  return renderWithProviders(<Comparison />, [ROUTE]);
}

describe("Comparison", () => {
  it("states the verdict as FAIL", async () => {
    serveComparison();
    renderComparison();

    const banner = await screen.findByRole("status", { name: "Regression verdict" });
    expect(within(banner).getByText("FAIL")).toBeInTheDocument();
  });

  it("lists every newly-failing case by id", async () => {
    serveComparison();
    renderComparison();

    expect(await screen.findByText("case-007")).toBeInTheDocument();
    expect(screen.getAllByText("case-019").length).toBeGreaterThan(0);
    expect(screen.getByText("case-042")).toBeInTheDocument();
    expect(screen.getByText(/3 cases newly failing/)).toBeInTheDocument();
  });

  it("lists newly-passing cases separately from regressions", async () => {
    serveComparison();
    renderComparison();

    expect(await screen.findByText("case-031")).toBeInTheDocument();
    expect(screen.getByText(/1 cases newly passing/)).toBeInTheDocument();
  });

  it("renders a badge, not a number, for a check with insufficient data", async () => {
    serveComparison();
    renderComparison();

    const row = (await screen.findByText("category.arithmetic.pass_rate")).closest("tr");
    expect(row).not.toBeNull();
    const cells = within(row as HTMLElement);

    expect(cells.getByText("insufficient samples")).toBeInTheDocument();
    expect(cells.getByText("insufficient data")).toBeInTheDocument();
    // The 30-point difference this check would otherwise have printed is absent.
    expect(row?.textContent).not.toContain("-30.0 pp");
    expect(row?.textContent).not.toContain("90.0%");
    expect(row?.textContent).not.toContain("60.0%");
    // The sample counts that make the badge interpretable are still shown.
    expect(row?.textContent).toContain("5 / 5");
  });

  it("shows the failing check's numbers with their intervals and threshold", async () => {
    serveComparison();
    renderComparison();

    const row = (await screen.findByText("pass_rate")).closest("tr");
    expect(row).not.toBeNull();
    expect(row?.textContent).toContain("86.0%");
    expect(row?.textContent).toContain("76.0%");
    expect(row?.textContent).toContain("-10.0 pp");
    expect(row?.textContent).toContain("max_absolute_decrease");
    expect(row?.textContent).toContain("(73.8-93.1%)");
  });

  it("caveats an overlapping interval without changing the verdict", async () => {
    serveComparison();
    renderComparison();

    const banner = await screen.findByRole("status", { name: "Regression verdict" });
    expect(within(banner).getByText("FAIL")).toBeInTheDocument();
    expect(
      within(banner).getByText("inconclusive: intervals overlap"),
    ).toBeInTheDocument();
  });

  it("names the cases whose content changed between the two runs", async () => {
    serveComparison();
    renderComparison();

    expect(await screen.findByText("case-changed-1")).toBeInTheDocument();
    expect(screen.getByText(/Changed case hash \(1\)/)).toBeInTheDocument();
  });

  it("asks for two runs when the URL names none", async () => {
    serveComparison();
    renderWithProviders(<Comparison />, ["/compare"]);

    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
  });
});
