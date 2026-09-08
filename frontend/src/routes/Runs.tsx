/**
 * Every stored run, filterable, with a two-click path into a comparison.
 *
 * Filters are pushed into the query string so a filtered listing is a URL
 * someone can paste into a ticket, and applied by the API rather than in the
 * browser so the result count is the real one. The three text filters are held
 * in local state and written to the URL after a pause, so typing a model name
 * issues one request rather than one per keystroke; the status dropdown, which
 * changes in one gesture, is written through immediately.
 */

/** How long a text filter waits after the last keystroke before it queries. */
const FILTER_DEBOUNCE_MS = 250;

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import type { RunStatus } from "../api/client";
import { useRuns } from "../api/queries";
import { NoRunsYet } from "../components/EmptyState";
import { Loading, QueryError } from "../components/QueryState";
import { RunTable } from "../components/RunTable";
import { formatCount, shortHash } from "../lib/format";

const PAGE_SIZE = 50;

const STATUSES: readonly RunStatus[] = [
  "pending",
  "running",
  "completed",
  "partial",
  "failed",
  "cancelled",
  "interrupted",
];

export function Runs() {
  const [params, setParams] = useSearchParams();
  const [baseline, setBaseline] = useState<string>("");
  const [candidate, setCandidate] = useState<string>("");

  const status = params.get("status") ?? "";
  const provider = params.get("provider") ?? "";
  const model = params.get("model") ?? "";
  const suite = params.get("suite") ?? "";
  const offset = Number(params.get("offset") ?? "0");

  // What the inputs show, which leads the URL by up to one debounce interval.
  const [draft, setDraft] = useState({ provider, model, suite });

  const query = useMemo(
    () => ({
      status: (status === "" ? undefined : status) as RunStatus | undefined,
      provider: provider === "" ? undefined : provider,
      model: model === "" ? undefined : model,
      suite: suite === "" ? undefined : suite,
      limit: PAGE_SIZE,
      offset: Number.isFinite(offset) ? offset : 0,
      order: "-created_at",
    }),
    [status, provider, model, suite, offset],
  );

  const runs = useRuns(query);

  const setParam = useCallback(
    (key: string, value: string) => {
      const next = new URLSearchParams(params);
      if (value === "") next.delete(key);
      else next.set(key, value);
      next.delete("offset");
      setParams(next, { replace: true });
    },
    [params, setParams],
  );

  // Keep the inputs in step when the URL changes from somewhere else, such as a
  // back navigation, without clobbering what is being typed.
  const committed = useRef({ provider, model, suite });
  useEffect(() => {
    if (
      committed.current.provider !== provider ||
      committed.current.model !== model ||
      committed.current.suite !== suite
    ) {
      committed.current = { provider, model, suite };
      setDraft({ provider, model, suite });
    }
  }, [provider, model, suite]);

  useEffect(() => {
    if (
      draft.provider === provider &&
      draft.model === model &&
      draft.suite === suite
    ) {
      return undefined;
    }
    const timer = setTimeout(() => {
      const next = new URLSearchParams(params);
      for (const [key, value] of Object.entries(draft)) {
        if (value === "") next.delete(key);
        else next.set(key, value);
      }
      next.delete("offset");
      committed.current = { ...draft };
      setParams(next, { replace: true });
    }, FILTER_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
    };
  }, [draft, provider, model, suite, params, setParams]);

  const page = (delta: number) => {
    const next = new URLSearchParams(params);
    const target = Math.max(0, offset + delta * PAGE_SIZE);
    if (target === 0) next.delete("offset");
    else next.set("offset", String(target));
    setParams(next, { replace: true });
  };

  const items = runs.data?.items ?? [];
  const total = runs.data?.total ?? 0;

  return (
    <>
      <div className="page-head">
        <h1>Runs</h1>
        <span className="dim mono">
          {formatCount(total)} stored, showing {formatCount(items.length)} from{" "}
          {formatCount(offset)}
        </span>
      </div>
      <p className="page-sub">
        Filters are applied by the API, so the count above describes the whole matching set
        and not just this page.
      </p>

      {runs.isError ? <QueryError what="runs" error={runs.error} /> : null}

      <div className="panel">
        <div className="panel-head">
          <div className="controls">
            <label htmlFor="filter-status" className="dim">
              status
            </label>
            <select
              id="filter-status"
              value={status}
              onChange={(event) => {
                setParam("status", event.target.value);
              }}
            >
              <option value="">any</option>
              {STATUSES.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>

            <label htmlFor="filter-provider" className="dim">
              provider
            </label>
            <input
              id="filter-provider"
              type="text"
              value={draft.provider}
              onChange={(event) => {
                const value = event.target.value;
                setDraft((current) => ({ ...current, provider: value }));
              }}
            />

            <label htmlFor="filter-model" className="dim">
              model
            </label>
            <input
              id="filter-model"
              type="text"
              value={draft.model}
              onChange={(event) => {
                const value = event.target.value;
                setDraft((current) => ({ ...current, model: value }));
              }}
            />

            <label htmlFor="filter-suite" className="dim">
              suite
            </label>
            <input
              id="filter-suite"
              type="text"
              value={draft.suite}
              onChange={(event) => {
                const value = event.target.value;
                setDraft((current) => ({ ...current, suite: value }));
              }}
            />
          </div>
          <span className="spacer" />
          <div className="controls">
            <button type="button" onClick={() => { page(-1); }} disabled={offset === 0}>
              previous
            </button>
            <button
              type="button"
              onClick={() => { page(1); }}
              disabled={offset + items.length >= total}
            >
              next
            </button>
          </div>
        </div>

        {runs.isPending ? (
          <Loading what="runs" />
        ) : items.length === 0 ? (
          <NoRunsYet />
        ) : (
          <RunTable runs={items} />
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Compare two runs</h2>
        </div>
        <div className="panel-body controls">
          <label htmlFor="compare-baseline" className="dim">
            baseline
          </label>
          <select
            id="compare-baseline"
            value={baseline}
            onChange={(event) => {
              setBaseline(event.target.value);
            }}
          >
            <option value="">choose a run</option>
            {items.map((run) => (
              <option key={run.id} value={run.id}>
                {run.label ?? shortHash(run.id)} · {run.model}
              </option>
            ))}
          </select>

          <label htmlFor="compare-candidate" className="dim">
            candidate
          </label>
          <select
            id="compare-candidate"
            value={candidate}
            onChange={(event) => {
              setCandidate(event.target.value);
            }}
          >
            <option value="">choose a run</option>
            {items.map((run) => (
              <option key={run.id} value={run.id}>
                {run.label ?? shortHash(run.id)} · {run.model}
              </option>
            ))}
          </select>

          {baseline === "" || candidate === "" || baseline === candidate ? (
            <span className="dim">pick two different runs</span>
          ) : (
            <Link
              to={`/compare?baseline=${encodeURIComponent(baseline)}&candidate=${encodeURIComponent(candidate)}`}
            >
              compare them
            </Link>
          )}
        </div>
      </div>
    </>
  );
}
