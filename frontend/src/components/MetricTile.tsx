/** A single headline number with its unit, its caveat and its supporting count. */

import type { ReactNode } from "react";

export interface MetricTileProps {
  label: string;
  value: ReactNode;
  /** The count, coverage note or caveat that makes the number interpretable. */
  sub?: ReactNode;
  title?: string;
}

export function MetricTile({ label, value, sub, title }: MetricTileProps) {
  return (
    <div className="tile" title={title}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub === undefined ? null : <div className="tile-sub">{sub}</div>}
    </div>
  );
}
