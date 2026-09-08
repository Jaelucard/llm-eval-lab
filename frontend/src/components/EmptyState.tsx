/**
 * The documented empty state.
 *
 * Each one names the exact command that produces the missing data, because a
 * blank panel that says "no data" is indistinguishable from a broken fetch.
 */

import type { ReactNode } from "react";

export interface EmptyStateProps {
  title: string;
  children: ReactNode;
}

export function EmptyState({ title, children }: EmptyStateProps) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      <div className="empty-body">{children}</div>
    </div>
  );
}

/** The empty state shown wherever a runs listing comes back with no rows. */
export function NoRunsYet() {
  return (
    <EmptyState title="No runs stored yet">
      Create one from the command line, then reload:{" "}
      <code>llm-eval run examples/benchmarks/smoke.yaml --provider fake --model fake-1</code>.
      The dashboard reads runs from the same database the CLI writes to.
    </EmptyState>
  );
}
