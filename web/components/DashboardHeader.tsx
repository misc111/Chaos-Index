"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { BET_STRATEGIES, getBetStrategyConfig, normalizeBetStrategy, type BetStrategy } from "@/lib/betting-strategy";
import { type LeagueCode, normalizeLeague, withLeague } from "@/lib/league";
import { isStaticStagingBuild } from "@/lib/static-staging";

const links: Array<[string, string]> = [
  ["/", "Overview"],
  ["/games-today", "Games Today"],
  ["/market-board", "Market Board"],
  ["/bet-history", "Bet History"],
  ["/actual-vs-expected", "Actual vs Expected"],
  ["/predictions", "Model Summary"],
];

function isActivePath(currentPath: string, href: string): boolean {
  const normalizedCurrentPath = currentPath !== "/" ? currentPath.replace(/\/+$/, "") : currentPath;
  const normalizedHref = href !== "/" ? href.replace(/\/+$/, "") : href;
  return normalizedCurrentPath === normalizedHref;
}

type RefreshResponse = {
  ok?: boolean;
  error?: string;
  details?: string;
  refreshed_at_utc?: string;
};

type DashboardTheme = "light" | "market-board-dark";

const DASHBOARD_THEME_KEY = "dashboard-theme";
const DARK_THEME: DashboardTheme = "market-board-dark";
const LIGHT_THEME: DashboardTheme = "light";

function isDashboardTheme(value: string | null): value is DashboardTheme {
  return value === DARK_THEME || value === LIGHT_THEME;
}

function applyDashboardTheme(theme: DashboardTheme): void {
  document.documentElement.setAttribute("data-dashboard-theme", theme);
}

function resolveDashboardTheme(): DashboardTheme {
  if (typeof document !== "undefined") {
    const currentTheme = document.documentElement.getAttribute("data-dashboard-theme");
    if (isDashboardTheme(currentTheme)) {
      return currentTheme;
    }
  }

  if (typeof window !== "undefined") {
    const storedTheme = window.localStorage.getItem(DASHBOARD_THEME_KEY);
    if (isDashboardTheme(storedTheme)) {
      return storedTheme;
    }
  }

  return DARK_THEME;
}

function hrefWithLeague(href: string, league: LeagueCode, searchParams: URLSearchParams): string {
  return hrefWithParams(href, searchParams, { league });
}

function hrefWithStrategy(href: string, strategy: BetStrategy, searchParams: URLSearchParams): string {
  return hrefWithParams(href, searchParams, { strategy });
}

function hrefWithParams(href: string, searchParams: URLSearchParams, updates: Record<string, string>): string {
  const params = new URLSearchParams(searchParams.toString());
  for (const [key, value] of Object.entries(updates)) {
    params.set(key, value);
  }
  const query = params.toString();
  return query ? `${href}?${query}` : href;
}

function formatRefreshTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString(undefined, {
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function HeaderFallback() {
  return (
    <>
      <h1 className="title app-title">NBA Win Probability Forecasting</h1>
      <div className="header-control-row">
        <div className="header-selection-stack">
          <div className="league-toggle-row" aria-label="League selection">
            <Link href="?league=NBA&strategy=balanced" className="league-toggle-btn active">
              NBA
            </Link>
            <Link href="?league=NHL&strategy=balanced" className="league-toggle-btn">
              NHL
            </Link>
          </div>
          <div className="strategy-toggle-stack">
            <span className="strategy-toggle-label">Bet Profile</span>
            <div className="strategy-toggle-row" aria-label="Bet strategy selection">
              <Link href="?league=NBA&strategy=balanced" className="strategy-toggle-btn active">
                <span className="strategy-toggle-title">Balanced</span>
                <span className="strategy-toggle-note">Standard sizing</span>
              </Link>
              <Link href="?league=NBA&strategy=riskAverse" className="strategy-toggle-btn">
                <span className="strategy-toggle-title">Risk Averse</span>
                <span className="strategy-toggle-note">Favorites only</span>
              </Link>
              <Link href="?league=NBA&strategy=riskLoving" className="strategy-toggle-btn">
                <span className="strategy-toggle-title">Risk Loving</span>
                <span className="strategy-toggle-note">Bigger swings</span>
              </Link>
            </div>
          </div>
        </div>
        <button type="button" className="theme-toggle-btn active" aria-pressed="true">
          <span className="theme-toggle-copy">
            <span className="theme-toggle-label">Dark Mode</span>
            <span className="theme-toggle-state">On</span>
          </span>
          <span className="theme-toggle-track" aria-hidden>
            <span className="theme-toggle-knob" />
          </span>
        </button>
      </div>
      <div className="refresh-row">
        <button type="button" className="refresh-btn" disabled>
          Refresh Data
        </button>
      </div>
        <div className="nav">
        {links.map(([href, label]) => (
          <Link href={`${href}?league=NBA&strategy=balanced`} key={href} className="nav-link">
            {label}
          </Link>
        ))}
      </div>
    </>
  );
}

function DashboardHeaderContent() {
  const pathname = usePathname() || "/";
  const searchParams = useSearchParams();
  const search = new URLSearchParams(searchParams.toString());
  const league = normalizeLeague(searchParams.get("league"));
  const strategy = normalizeBetStrategy(searchParams.get("strategy"));
  const strategyConfig = getBetStrategyConfig(strategy);
  const staticStaging = isStaticStagingBuild();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState("");
  const [theme, setTheme] = useState<DashboardTheme>(DARK_THEME);
  const refreshedAtRaw = searchParams.get("refreshedAt");
  const refreshedLeague = searchParams.get("refreshedLeague");

  const showRefreshedStamp =
    typeof refreshedAtRaw === "string" && refreshedAtRaw.length > 0 && refreshedLeague === league;

  const refreshedAtLabel = showRefreshedStamp ? formatRefreshTimestamp(refreshedAtRaw) : "";
  const isDarkTheme = theme === DARK_THEME;

  useEffect(() => {
    const resolvedTheme = resolveDashboardTheme();
    applyDashboardTheme(resolvedTheme);
    setTheme(resolvedTheme);
  }, []);

  const handleRefresh = async () => {
    if (staticStaging) {
      return;
    }

    setIsRefreshing(true);
    setRefreshError("");

    try {
      const response = await fetch(withLeague("/api/refresh-data", league), {
        method: "POST",
      });
      const payload = (await response.json().catch(() => ({}))) as RefreshResponse;
      if (!response.ok || payload.ok === false) {
        const detailSuffix = payload.details ? ` ${payload.details}` : "";
        throw new Error(payload.error ? `${payload.error}${detailSuffix}` : `Refresh failed (${response.status}).`);
      }

      const refreshedAtUtc = payload.refreshed_at_utc || new Date().toISOString();
      const nextSearch = new URLSearchParams(search.toString());
      nextSearch.set("league", league);
      nextSearch.set("refreshedAt", refreshedAtUtc);
      nextSearch.set("refreshedLeague", league);
      window.location.assign(`${pathname}?${nextSearch.toString()}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to refresh data right now.";
      setRefreshError(message.length > 300 ? `${message.slice(0, 297)}...` : message);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleThemeToggle = () => {
    const nextTheme = isDarkTheme ? LIGHT_THEME : DARK_THEME;
    applyDashboardTheme(nextTheme);
    setTheme(nextTheme);
    try {
      window.localStorage.setItem(DASHBOARD_THEME_KEY, nextTheme);
    } catch {
      // Ignore persistence failures and keep the in-memory toggle responsive.
    }
  };

  return (
    <>
      <h1 className="title app-title">{league} Win Probability Forecasting</h1>

      <div className="header-control-row">
        <div className="header-selection-stack">
          <div className="league-toggle-row" aria-label="League selection">
            {(["NBA", "NHL"] as LeagueCode[]).map((code) => (
              <Link
                href={hrefWithLeague(pathname, code, search)}
                key={code}
                className={`league-toggle-btn ${league === code ? "active" : ""}`}
              >
                {code}
              </Link>
            ))}
          </div>
          <div className="strategy-toggle-stack">
            <span className="strategy-toggle-label">Bet Profile</span>
            <div className="strategy-toggle-row" aria-label="Bet strategy selection">
              {BET_STRATEGIES.map((code) => (
                <Link
                  href={hrefWithStrategy(pathname, code, search)}
                  key={code}
                  className={`strategy-toggle-btn ${strategy === code ? "active" : ""}`}
                >
                  <span className="strategy-toggle-title">{getBetStrategyConfig(code).label}</span>
                  <span className="strategy-toggle-note">{getBetStrategyConfig(code).shortLabel}</span>
                </Link>
              ))}
            </div>
          </div>
        </div>
        <button
          type="button"
          className={`theme-toggle-btn ${isDarkTheme ? "active" : ""}`}
          onClick={handleThemeToggle}
          aria-pressed={isDarkTheme}
          aria-label={isDarkTheme ? "Switch to light mode" : "Switch to dark mode"}
        >
          <span className="theme-toggle-copy">
            <span className="theme-toggle-label">Dark Mode</span>
            <span className="theme-toggle-state">{isDarkTheme ? "On" : "Off"}</span>
          </span>
          <span className="theme-toggle-track" aria-hidden>
            <span className="theme-toggle-knob" />
          </span>
        </button>
      </div>

      <div className="refresh-row">
        <button
          type="button"
          className="refresh-btn"
          onClick={handleRefresh}
          disabled={isRefreshing || staticStaging}
          aria-busy={isRefreshing}
        >
          {staticStaging ? (
            "Snapshot Only"
          ) : isRefreshing ? (
            <>
              <span className="refresh-spinner" aria-hidden />
              Refreshing {league}...
            </>
          ) : (
            "Refresh Data"
          )}
        </button>
        <div className="refresh-meta" aria-live="polite">
          <p className="small">Active bet profile: {strategyConfig.label}. {strategyConfig.description}</p>
          {staticStaging ? <p className="small">GitHub Pages staging uses committed snapshot data.</p> : null}
          {!staticStaging ? (
            <p className="small">Ingest only. No feature rebuild and no retraining.</p>
          ) : null}
          {isRefreshing ? (
            <>
              <p className="small">Refreshing {league} data without rebuilding models...</p>
              <div className="refresh-progress-track">
                <span className="refresh-progress-fill" />
              </div>
            </>
          ) : null}
          {!isRefreshing && showRefreshedStamp ? (
            <p className="small">Data refreshed as of {refreshedAtLabel}</p>
          ) : null}
          {refreshError ? <p className="small refresh-error">{refreshError}</p> : null}
        </div>
      </div>

      <div className="nav">
        {links.map(([href, label]) => {
          const isActive = isActivePath(pathname, href);
          return (
            <Link
              href={hrefWithLeague(href, league, search)}
              key={href}
              className={`nav-link ${isActive ? "active" : ""}`}
              aria-current={isActive ? "page" : undefined}
            >
              {label}
            </Link>
          );
        })}
      </div>
    </>
  );
}

export default function DashboardHeader() {
  return (
    <Suspense fallback={<HeaderFallback />}>
      <DashboardHeaderContent />
    </Suspense>
  );
}
