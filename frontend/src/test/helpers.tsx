/**
 * Test helpers: a fresh query client per test and a stubbed `fetch`.
 *
 * Routes are rendered inside a real router and a real QueryClientProvider so a
 * test exercises the same data path the browser does. Only the network is
 * replaced.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi } from "vitest";

/** A route handler: given a pathname and its search params, return a body. */
export type RouteHandler = (url: URL) => unknown;

/** Install a `fetch` that answers from a path-prefix routing table. */
export function stubFetch(routes: Record<string, RouteHandler>): void {
  const handler = vi.fn((input: RequestInfo | URL) => {
    const raw =
      typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const url = new URL(raw, "http://localhost");
    const match = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((prefix) => url.pathname === prefix || url.pathname.startsWith(`${prefix}/`));

    if (match === undefined) {
      return Promise.resolve(
        new Response(
          JSON.stringify({ title: "Not Found", detail: raw, status: 404 }),
          { status: 404, headers: { "content-type": "application/problem+json" } },
        ),
      );
    }

    const body = routes[match]?.(url);
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  });
  vi.stubGlobal("fetch", handler);
}

/** Render a component inside the providers the app supplies in production. */
export function renderWithProviders(
  ui: ReactElement,
  initialEntries: string[] = ["/"],
): RenderResult {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={initialEntries}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

/**
 * Render a component that reads route parameters, mounted at a real path
 * pattern so `useParams` resolves the way it does in the running app.
 */
export function renderAtRoute(
  pattern: string,
  ui: ReactElement,
  entry: string,
): RenderResult {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[entry]}>
        <Routes>
          <Route path={pattern} element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}
