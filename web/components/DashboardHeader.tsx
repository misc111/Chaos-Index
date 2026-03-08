"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import {
  BET_SIZING_STYLES,
  BET_STRATEGIES,
  getBetSizingStyleConfig,
  getBetStrategyConfig,
  normalizeBetSizingStyle,
  normalizeBetStrategy,
  type BetSizingStyle,
  type BetStrategy,
} from "@/lib/betting-strategy";
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

const DEFAULT_QUERY = "?league=NBA&strategy=balanced&sizingStyle=continuous";

type RefreshResponse = {
  ok?: boolean;
  error?: string;
  details?: string;
  refreshed_at_utc?: string;
};

type DashboardTheme = "light" | "market-board-dark";

type SidebarControlsProps = {
  isRefreshing: boolean;
  league: LeagueCode;
  onRefresh?: () => void;
  onThemeToggle?: () => void;
  pathname: string;
  refreshError: string;
  refreshedAtLabel: string;
  search: URLSearchParams;
  showRefreshedStamp: boolean;
  sizingStyle: BetSizingStyle;
  staticStaging: boolean;
  strategy: BetStrategy;
  theme: DashboardTheme;
};

const DASHBOARD_THEME_KEY = "dashboard-theme";
const DARK_THEME: DashboardTheme = "market-board-dark";
const LIGHT_THEME: DashboardTheme = "light";

function isActivePath(currentPath: string, href: string): boolean {
  const normalizedCurrentPath = currentPath !== "/" ? currentPath.replace(/\/+$/, "") : currentPath;
  const normalizedHref = href !== "/" ? href.replace(/\/+$/, "") : href;
  return normalizedCurrentPath === normalizedHref;
}

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

function hrefWithSizingStyle(href: string, sizingStyle: BetSizingStyle, searchParams: URLSearchParams): string {
  return hrefWithParams(href, searchParams, { sizingStyle });
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

function SidebarControls({
  isRefreshing,
  league,
  onRefresh,
  onThemeToggle,
  pathname,
  refreshError,
  refreshedAtLabel,
  search,
  showRefreshedStamp,
  sizingStyle,
  staticStaging,
  strategy,
  theme,
}: SidebarControlsProps) {
  const isDarkTheme = theme === DARK_THEME;
  const strategyConfig = getBetStrategyConfig(strategy);
  const sizingStyleConfig = getBetSizingStyleConfig(sizingStyle);

  return (
    <>
      <div className="sidebar-brand card">
        <p className="sidebar-eyebrow">Control Center</p>
        <h2 className="sidebar-title">Dashboard Inputs</h2>
        <p className="small sidebar-copy">
          Switch leagues and stake assumptions without leaving the current view.
        </p>
      </div>

      <section className="sidebar-card card" aria-labelledby="sidebar-league-title">
        <span className="strategy-toggle-label" id="sidebar-league-title">
          League
        </span>
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
      </section>

      <section className="sidebar-card card" aria-labelledby="sidebar-profile-title">
        <span className="strategy-toggle-label" id="sidebar-profile-title">
          Bet Profile
        </span>
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
      </section>

      <section className="sidebar-card card" aria-labelledby="sidebar-sizing-title">
        <span className="strategy-toggle-label" id="sidebar-sizing-title">
          Amount Bet
        </span>
        <div className="strategy-toggle-row" aria-label="Bet sizing style selection">
          {BET_SIZING_STYLES.map((code) => (
            <Link
              href={hrefWithSizingStyle(pathname, code, search)}
              key={code}
              className={`strategy-toggle-btn ${sizingStyle === code ? "active" : ""}`}
            >
              <span className="strategy-toggle-title">{getBetSizingStyleConfig(code).label}</span>
              <span className="strategy-toggle-note">{getBetSizingStyleConfig(code).shortLabel}</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="sidebar-card card" aria-labelledby="sidebar-utilities-title">
        <span className="strategy-toggle-label" id="sidebar-utilities-title">
          Utilities
        </span>
        <div className="refresh-row">
          <button
            type="button"
            className="refresh-btn"
            onClick={onRefresh}
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
            <p className="small">
              Active bet profile: {strategyConfig.label}. {strategyConfig.description}
            </p>
            <p className="small">
              Amount bet mode: {sizingStyleConfig.label}. {sizingStyleConfig.description}
            </p>
            {staticStaging ? <p className="small">GitHub Pages staging uses committed snapshot data.</p> : null}
            {!staticStaging ? <p className="small">Ingest only. No feature rebuild and no retraining.</p> : null}
            {isRefreshing ? (
              <>
                <p className="small">Refreshing {league} data without rebuilding models...</p>
                <div className="refresh-progress-track">
                  <span className="refresh-progress-fill" />
                </div>
              </>
            ) : null}
            {!isRefreshing && showRefreshedStamp ? <p className="small">Data refreshed as of {refreshedAtLabel}</p> : null}
            {refreshError ? <p className="small refresh-error">{refreshError}</p> : null}
          </div>
        </div>

        <button
          type="button"
          className={`theme-toggle-btn ${isDarkTheme ? "active" : ""}`}
          onClick={onThemeToggle}
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
      </section>
    </>
  );
}

function DashboardSidebarFallback() {
  return (
    <aside className="dashboard-sidebar" aria-label="Dashboard controls">
      <div className="dashboard-sidebar-inner">
        <SidebarControls
          isRefreshing={false}
          league="NBA"
          pathname="/"
          refreshError=""
          refreshedAtLabel=""
          search={new URLSearchParams("league=NBA&strategy=balanced&sizingStyle=continuous")}
          showRefreshedStamp={false}
          sizingStyle="continuous"
          staticStaging={false}
          strategy="balanced"
          theme={DARK_THEME}
        />
      </div>
    </aside>
  );
}

function DashboardSidebarContent() {
  const pathname = usePathname() || "/";
  const searchParams = useSearchParams();
  const search = new URLSearchParams(searchParams.toString());
  const league = normalizeLeague(searchParams.get("league"));
  const strategy = normalizeBetStrategy(searchParams.get("strategy"));
  const sizingStyle = normalizeBetSizingStyle(searchParams.get("sizingStyle"));
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
    <aside className="dashboard-sidebar" aria-label="Dashboard controls">
      <div className="dashboard-sidebar-inner">
        <SidebarControls
          isRefreshing={isRefreshing}
          league={league}
          onRefresh={handleRefresh}
          onThemeToggle={handleThemeToggle}
          pathname={pathname}
          refreshError={refreshError}
          refreshedAtLabel={refreshedAtLabel}
          search={search}
          showRefreshedStamp={showRefreshedStamp}
          sizingStyle={sizingStyle}
          staticStaging={staticStaging}
          strategy={strategy}
          theme={theme}
        />
      </div>
    </aside>
  );
}

function HeaderFallback() {
  return (
    <>
      <div className="dashboard-topbar">
        <div className="dashboard-title-block">
          <p className="sidebar-eyebrow dashboard-eyebrow">Chaos Index</p>
          <h1 className="title app-title">NBA Win Probability Forecasting</h1>
          <p className="small dashboard-subtitle">
            Independent win probabilities you can compare against the market.
          </p>
        </div>
      </div>

      <nav className="nav dashboard-nav" aria-label="Primary dashboard navigation">
        {links.map(([href, label]) => (
          <Link href={`${href}${DEFAULT_QUERY}`} key={href} className="nav-link">
            {label}
          </Link>
        ))}
      </nav>
    </>
  );
}

function DashboardHeaderContent() {
  const pathname = usePathname() || "/";
  const searchParams = useSearchParams();
  const search = new URLSearchParams(searchParams.toString());
  const league = normalizeLeague(searchParams.get("league"));

  return (
    <>
      <div className="dashboard-topbar">
        <div className="dashboard-title-block">
          <p className="sidebar-eyebrow dashboard-eyebrow">Chaos Index</p>
          <h1 className="title app-title">{league} Win Probability Forecasting</h1>
          <p className="small dashboard-subtitle">
            Independent win probabilities you can compare against the market.
          </p>
        </div>
      </div>

      <nav className="nav dashboard-nav" aria-label="Primary dashboard navigation">
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
      </nav>
    </>
  );
}

export function DashboardSidebar() {
  return (
    <Suspense fallback={<DashboardSidebarFallback />}>
      <DashboardSidebarContent />
    </Suspense>
  );
}

export default function DashboardHeader() {
  return (
    <Suspense fallback={<HeaderFallback />}>
      <DashboardHeaderContent />
    </Suspense>
  );
}
