/**
 * Loading and failure rendering, in one place.
 *
 * A failed fetch says what failed and what the API said about it. The RFC 9457
 * problem detail the backend returns already carries a usable title and detail,
 * so it is shown rather than replaced with a generic apology.
 */

import { ApiError, isAuthRequired } from "../api/client";

export function Loading({ what }: { what: string }) {
  return (
    <div className="loading" role="status">
      loading {what}...
    </div>
  );
}

/**
 * Why a 401 gets its own copy instead of the generic problem-detail message.
 *
 * Token-protected mode is API-only by design: the bundled dashboard never
 * holds or sends the bearer token, so telling the reader "401 Unauthorized"
 * would be technically true and practically useless. See `README.md#the-dashboard`.
 */
export const AUTH_REQUIRED_MESSAGE =
  "This server requires a bearer token (LLM_EVAL_API_TOKEN is set). The " +
  "dashboard does not send tokens by design: token-protected mode is " +
  "API-only. Run the server on loopback and open it locally or through an " +
  "SSH tunnel (ssh -L 8000:127.0.0.1:8000 host), or call the API directly " +
  "with Authorization: Bearer <token>.";

/** A single, app-wide notice for when the dashboard is running against a token-guarded API. */
export function AuthRequiredBanner({ error }: { error: unknown }) {
  if (!isAuthRequired(error)) return null;
  return (
    <div className="error-box" role="alert">
      <strong>Bearer token required.</strong> {AUTH_REQUIRED_MESSAGE}
    </div>
  );
}

export interface QueryErrorProps {
  what: string;
  error: unknown;
}

export function QueryError({ what, error }: QueryErrorProps) {
  if (isAuthRequired(error)) {
    return (
      <div className="error-box" role="alert">
        <strong>Could not load {what}.</strong> {AUTH_REQUIRED_MESSAGE}
      </div>
    );
  }

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
