/**
 * Token-protected API mode: the dashboard never sends a bearer token (see
 * `README.md#the-dashboard`), so a 401 from a guarded endpoint must render the
 * specific "token mode is API-only" explanation, not the generic problem-detail
 * message.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError, isAuthRequired } from "../api/client";
import { App } from "../App";
import { AUTH_REQUIRED_MESSAGE, QueryError } from "../components/QueryState";
import { health } from "./fixtures";
import { renderWithProviders } from "./helpers";

const UNAUTHORIZED = {
  type: "/problems/unauthorized",
  title: "Unauthorized",
  detail: "A bearer token is required. Send Authorization: Bearer <token>.",
  status: 401,
};

/**
 * Install a fetch stub where each route answers with its own status code.
 *
 * `stubFetch` in `./helpers` always answers 200, which cannot express a
 * guarded route returning 401, so this is a local variant for that one case.
 */
function stubFetchWithStatus(
  routes: Record<string, { status: number; body: unknown }>,
): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const raw =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      const url = new URL(raw, "http://localhost");
      const match = Object.keys(routes)
        .sort((a, b) => b.length - a.length)
        .find((prefix) => url.pathname === prefix || url.pathname.startsWith(`${prefix}/`));

      if (match === undefined) {
        return Promise.resolve(
          new Response(JSON.stringify({ title: "Not Found", detail: raw, status: 404 }), {
            status: 404,
            headers: { "content-type": "application/problem+json" },
          }),
        );
      }

      const route = routes[match];
      if (route === undefined) {
        throw new Error(`unreachable: matched key ${match} missing from routes`);
      }
      return Promise.resolve(
        new Response(JSON.stringify(route.body), {
          status: route.status,
          headers: {
            "content-type":
              route.status >= 400 ? "application/problem+json" : "application/json",
          },
        }),
      );
    }),
  );
}

describe("isAuthRequired", () => {
  it("is true only for a 401 ApiError", () => {
    expect(isAuthRequired(new ApiError(401, "nope", UNAUTHORIZED))).toBe(true);
    expect(isAuthRequired(new ApiError(404, "not found", null))).toBe(false);
    expect(isAuthRequired(new ApiError(500, "boom", null))).toBe(false);
    expect(isAuthRequired(new Error("network down"))).toBe(false);
    expect(isAuthRequired("nope")).toBe(false);
    expect(isAuthRequired(undefined)).toBe(false);
  });
});

describe("token-protected mode", () => {
  it("shows the API-only explanation as an app banner when /api/version 401s", async () => {
    stubFetchWithStatus({
      "/api/health": { status: 200, body: health },
      "/api/version": { status: 401, body: UNAUTHORIZED },
      "/api/runs": { status: 401, body: UNAUTHORIZED },
      "/api/models/summary": { status: 401, body: UNAUTHORIZED },
    });

    renderWithProviders(<App />, ["/"]);

    await waitFor(() => {
      expect(screen.getByText(/Bearer token required\./)).toBeInTheDocument();
    });
    // The same explanation can also appear per-panel via `QueryError` for any
    // other guarded query that 401s (e.g. the runs list on this route), so
    // this asserts the text exists at all rather than requiring one instance.
    expect(screen.getAllByText(/token-protected mode is API-only/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/LLM_EVAL_API_TOKEN/).length).toBeGreaterThan(0);
  });

  it("does not show the banner when the API answers without a token requirement", async () => {
    stubFetchWithStatus({
      "/api/health": { status: 200, body: health },
      "/api/version": {
        status: 200,
        body: { version: "0.1.0", python: "3.13.1", schema_version: 1 },
      },
      "/api/runs": {
        status: 200,
        body: { items: [], total: 0, limit: 50, offset: 0 },
      },
      "/api/models/summary": { status: 200, body: [] },
    });

    renderWithProviders(<App />, ["/"]);

    await waitFor(() => {
      expect(screen.getByText(/api: ok/)).toBeInTheDocument();
    });
    expect(screen.queryByText(/Bearer token required\./)).not.toBeInTheDocument();
  });

  it("QueryError shows the API-only explanation for a 401", () => {
    render(<QueryError what="runs" error={new ApiError(401, "401 Unauthorized", UNAUTHORIZED)} />);
    expect(screen.getByText(/Could not load runs\./)).toBeInTheDocument();
    expect(screen.getByText(AUTH_REQUIRED_MESSAGE)).toBeInTheDocument();
  });

  it("QueryError still shows the generic message for a non-401 error", () => {
    render(<QueryError what="runs" error={new ApiError(500, "500 boom", null)} />);
    expect(screen.getByText(/Could not load runs\./)).toBeInTheDocument();
    expect(screen.getByText(/500 boom/)).toBeInTheDocument();
    expect(screen.queryByText(/token-protected mode is API-only/i)).not.toBeInTheDocument();
  });
});
