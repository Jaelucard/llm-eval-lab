/**
 * A scroll container a keyboard can actually reach.
 *
 * Wide tables and the two charts overflow horizontally at laptop width, and a
 * `div` with `overflow-x: auto` is not focusable, so a keyboard-only reader
 * cannot scroll it at all. Giving each container `tabindex="0"`, a `region`
 * role and a name makes it a stop in the tab order and announces what is inside
 * it; the `:focus-visible` ring in the stylesheet makes the stop visible.
 */

import type { ReactNode } from "react";

export interface ScrollRegionProps {
  /** Accessible name, e.g. "Recent runs table". */
  label: string;
  /** `table` for a horizontally scrolling table, `chart` for an SVG frame. */
  variant?: "table" | "chart";
  children: ReactNode;
}

export function ScrollRegion({ label, variant = "table", children }: ScrollRegionProps) {
  return (
    <div
      className={variant === "chart" ? "chart-frame" : "table-scroll"}
      role="region"
      aria-label={label}
      tabIndex={0}
    >
      {children}
    </div>
  );
}
