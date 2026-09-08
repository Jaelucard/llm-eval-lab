/**
 * The two primitives the presentation rules are enforced in.
 *
 * `ConfidenceInterval` is the funnel every rate passes through, so its
 * behaviour when the API supplied no interval and when it supplied no sample
 * count is what decides whether a bare number can reach the screen at all.
 * `OutputBlock` and `ScrollRegion` carry the keyboard access for every
 * overflowing region in the app.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ConfidenceInterval } from "../components/ConfidenceInterval";
import { OutputBlock } from "../components/OutputBlock";
import { ScrollRegion } from "../components/ScrollRegion";

describe("ConfidenceInterval", () => {
  it("prints the interval the API supplied, beside the estimate and its n", () => {
    const { container } = render(
      <ConfidenceInterval rate={0.8} interval={[0.7288, 0.8548]} n={150} />,
    );
    expect(screen.getByText("80.0%")).toBeInTheDocument();
    expect(screen.getByText("(72.9-85.5%)")).toBeInTheDocument();
    expect(screen.getByText("n=150")).toBeInTheDocument();
    expect(container.querySelector(".ci-missing")).toBeNull();
  });

  it("says an interval is absent rather than leaving the number bare", () => {
    const { container } = render(<ConfidenceInterval rate={0.8} interval={null} n={150} />);
    expect(screen.getByText("(no interval)")).toBeInTheDocument();
    expect(container.querySelector(".ci")).toBeNull();
  });

  it("qualifies the sample count when it is not the rate's own denominator", () => {
    render(
      <ConfidenceInterval rate={0.8} interval={null} n={50} nLabel="completed" />,
    );
    expect(screen.getByText("n=50 completed")).toBeInTheDocument();
  });

  it("marks an unknown sample count instead of omitting it silently", () => {
    render(<ConfidenceInterval rate={0.8} interval={null} n={null} />);
    expect(screen.getByText("n unknown")).toBeInTheDocument();
  });

  it("badges a rate resting on fewer cases than its floor", () => {
    render(<ConfidenceInterval rate={0.75} interval={[0.47, 0.91]} n={12} />);
    const badge = screen.getByText("insufficient samples");
    expect(badge).toHaveAttribute(
      "title",
      "n=12 is below the 20-sample floor for this statistic. A benchmark this size cannot separate small differences from noise.",
    );
    // The estimate is still shown; the badge sits beside it rather than hiding it.
    expect(screen.getByText("75.0%")).toBeInTheDocument();
  });

  it("caveats an overlap with the interval it was asked to compare against", () => {
    render(
      <ConfidenceInterval
        rate={0.76}
        interval={[0.62, 0.86]}
        n={50}
        compareWith={[0.74, 0.93]}
      />,
    );
    expect(screen.getByText("inconclusive: intervals overlap")).toBeInTheDocument();
  });
});

describe("keyboard-reachable regions", () => {
  it("makes a scroll region a focus stop with a name", () => {
    render(
      <ScrollRegion label="Example table">
        <table>
          <tbody>
            <tr>
              <td>cell</td>
            </tr>
          </tbody>
        </table>
      </ScrollRegion>,
    );
    const region = screen.getByRole("region", { name: "Example table" });
    expect(region).toHaveAttribute("tabindex", "0");
    expect(region).toHaveClass("table-scroll");
  });

  it("uses the chart frame class for a chart region", () => {
    render(
      <ScrollRegion label="Example chart" variant="chart">
        <svg />
      </ScrollRegion>,
    );
    expect(screen.getByRole("region", { name: "Example chart" })).toHaveClass("chart-frame");
  });

  it("makes a model output block scrollable from the keyboard", () => {
    render(<OutputBlock label="Model output" text={"line one\nline two"} />);
    const block = screen.getByRole("region", { name: "Model output" });
    expect(block.tagName).toBe("PRE");
    expect(block).toHaveAttribute("tabindex", "0");
    // Untrusted text is a text child, never markup.
    expect(block.textContent).toBe("line one\nline two");
  });

  it("renders angle brackets in model output as text, not as elements", () => {
    render(<OutputBlock label="Model output" text={'<img src=x onerror="boom">'} />);
    const block = screen.getByRole("region", { name: "Model output" });
    expect(block.querySelector("img")).toBeNull();
    expect(block.textContent).toBe('<img src=x onerror="boom">');
  });
});
