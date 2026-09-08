/**
 * Loading and failure rendering, in one place.
 *
 * A failed fetch says what failed and what the API said about it. The RFC 9457
 * problem detail the backend returns already carries a usable title and detail,
 * so it is shown rather than replaced with a generic apology.
 */

import { ApiError } from "../api/client";

export function Loading({ what }: { what: string }) {
  return (
    <div className="loading" role="status">
      loading {what}...
    </div>
  );
}

export interface QueryErrorProps {
  what: string;
  error: unknown;
}

export function QueryError({ what, error }: QueryErrorProps) {
  let message: string;
  if (error instanceof ApiError) {
    message = error.problem
      ? `${String(error.status)} ${error.problem.title}: ${error.problem.detail}`
      : error.message;
  } else if (error instanceof Error) {
    message = error.message;
  } else {
    message = String(error);
  }

  return (
    <div className="error-box" role="alert">
      <strong>Could not load {what}.</strong> {message}
    </div>
  );
}
