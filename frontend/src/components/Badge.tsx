/** The one badge primitive. Every status word in the dashboard is one of these. */

import type { ReactNode } from "react";

export type BadgeTone = "pass" | "fail" | "warn" | "muted" | "accent";

export interface BadgeProps {
  tone: BadgeTone;
  children: ReactNode;
  title?: string | undefined;
}

export function Badge({ tone, children, title }: BadgeProps) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

/** A dashed advisory note that sits beside a badge without overriding it. */
export function Caveat({
  children,
  title,
}: {
  children: ReactNode;
  title?: string | undefined;
}) {
  return (
    <span className="caveat" title={title}>
      {children}
    </span>
  );
}
