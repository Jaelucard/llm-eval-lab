/** Application entry point: mount React under a query client and a router. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import { App } from "./App";
import "./styles.css";

/**
 * Retries are off for everything except transient network failures, which
 * TanStack already distinguishes. This dashboard talks to a local process: a
 * 404 for a run id will still be a 404 three attempts later, and retrying it
 * only delays the message the reader needs.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const container = document.getElementById("root");
if (container === null) {
  throw new Error("The #root element is missing from index.html.");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
