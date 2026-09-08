/**
 * Application shell: the header, the theme toggle and the route table.
 *
 * The routes are real paths rather than hashes because the backend's static
 * mount falls back to `index.html` for anything that is not an API path or a
 * bundled file, so a deep link into a run works in production as well as in the
 * dev server.
 */

import { useCallback, useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { useHealth, useVersion } from "./api/queries";
import { Comparison } from "./routes/Comparison";
import { ModelComparison } from "./routes/ModelComparison";
import { Overview } from "./routes/Overview";
import { RunDetail } from "./routes/RunDetail";
import { Runs } from "./routes/Runs";

type Theme = "light" | "dark" | "system";

const THEME_KEY = "llm-eval-lab.theme";

function readStoredTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // Storage can be unavailable; the system theme is a fine answer.
  }
  return "system";
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readStoredTheme);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", theme);
    }
    try {
      if (theme === "system") window.localStorage.removeItem(THEME_KEY);
      else window.localStorage.setItem(THEME_KEY, theme);
    } catch {
      // A theme that does not persist is still a theme.
    }
  }, [theme]);

  const cycle = useCallback(() => {
    setTheme((current) =>
      current === "system" ? "light" : current === "light" ? "dark" : "system",
    );
  }, []);

  return (
    <button type="button" onClick={cycle} title="Cycle theme: system, light, dark">
      theme: {theme}
    </button>
  );
}

function HeaderStatus() {
  const health = useHealth();
  const version = useVersion();

  if (health.isError) {
    return <span title="The dashboard could not reach the API.">api: unreachable</span>;
  }
  if (!health.data) {
    return <span>api: checking</span>;
  }
  return (
    <span title={`Database: ${health.data.database}`}>
      api: {health.data.status} · v{version.data?.version ?? health.data.version}
    </span>
  );
}

export function App() {
  return (
    <>
      <header className="app-header">
        <span className="brand">llm-eval-lab</span>
        <nav className="nav">
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/compare">Comparison</NavLink>
          <NavLink to="/models">Models</NavLink>
        </nav>
        <div className="header-right">
          <HeaderStatus />
          <ThemeToggle />
        </div>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
          <Route path="/compare" element={<Comparison />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </>
  );
}
