"use client";

import type { CSSProperties } from "react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import {
  BET_STRATEGIES,
  getDefaultBetStrategyForLeague,
  getBetStrategyConfig,
  normalizeBetStrategy,
  type BetStrategy,
} from "@/lib/betting-strategy";
import { ALL_LEAGUES, displayLeagueLabel, type LeagueCode, normalizeLeague, withLeague } from "@/lib/league";
import { isStaticStagingBuild } from "@/lib/static-staging";

const links: Array<[string, string]> = [
  ["/", "Overview"],
  ["/games-today", "Games Today"],
  ["/market-board", "Market Board"],
  ["/research-desk", "Research Desk"],
  ["/meet-the-models", "Meet Models"],
  ["/bet-history", "Bet History"],
  ["/actual-vs-expected", "Actual vs Expected"],
  ["/predictions", "Model Summary"],
  ["/performance", "Performance"],
  ["/bet-sizing", "Bet Sizing"],
];

const DEFAULT_QUERY = `?league=MLB&strategy=${getDefaultBetStrategyForLeague("MLB")}`;
const SHOW_LEAGUE_SELECTOR = ALL_LEAGUES.length > 1;
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
type RefreshResponse = {
  ok?: boolean;
  error?: string;
  details?: string;
  refreshed_at_utc?: string;
};

type DashboardTheme = "light" | "market-board-dark";

type RouteHero = {
  title: string;
  copy: string;
  image: string;
  signals: [string, string, string];
};

const routeHeroes: Record<string, RouteHero> = {
  "/actual-vs-expected": {
    title: "Actual vs Expected",
    copy: "A clean check on whether pregame probabilities are meeting real MLB outcomes.",
    image: "/images/pages/actual-vs-expected.png",
    signals: ["Expected path", "Actual result", "Pregame proof"],
  },
  "/bet-history": {
    title: "Bet History",
    copy: "Replay the ledger, bankroll path, and realized betting outcomes without the noise.",
    image: "/images/pages/bet-history.png",
    signals: ["Replay ledger", "Bankroll path", "Settled results"],
  },
  "/bet-sizing": {
    title: "Bet Sizing",
    copy: "Turn edge into stake size with budget, caps, and uncertainty kept visible.",
    image: "/images/pages/bet-sizing.png",
    signals: ["Risk budget", "Stake cap", "Bankroll scale"],
  },
  "/calibration": {
    title: "Calibration",
    copy: "Reliability curves, error checks, and probability discipline for the model lane.",
    image: "/images/pages/calibration.png",
    signals: ["ECE", "MCE", "Reliability"],
  },
  "/diagnostics": {
    title: "Diagnostics",
    copy: "Coefficient, residual, and stability checks for the GLM-family model program.",
    image: "/images/pages/diagnostics.png",
    signals: ["Residuals", "Coefficients", "Stability"],
  },
  "/games-today": {
    title: "Games Today",
    copy: "Today’s MLB slate, first-pitch timing, win probabilities, and bet calls.",
    image: "/images/pages/games-today.png",
    signals: ["Slate", "First pitch", "Bet call"],
  },
  "/leaderboard": {
    title: "Leaderboard",
    copy: "Model ranks, rolling performance, and the current evidence order.",
    image: "/images/pages/leaderboard.png",
    signals: ["Rank", "Lift", "Evidence"],
  },
  "/market-board": {
    title: "Market Board",
    copy: "A market-aware pricing board for MLB moneyline, spread, total, and model edge.",
    image: "/images/pages/market-board.png",
    signals: ["Odds", "Fair price", "Model edge"],
  },
  "/meet-the-models": {
    title: "Meet the Models",
    copy: "A plain-English roster of the model family, what each one sees, and how each one can get fooled.",
    image: "/images/pages/predictions.png",
    signals: ["Plain English", "Model roles", "Trust notes"],
  },
  "/nested-tournament": {
    title: "Nested Tournament",
    copy: "Target-scoped model-family contests, challengers, and champion evidence.",
    image: "/images/pages/nested-tournament.png",
    signals: ["Family lanes", "Challengers", "Champion"],
  },
  "/performance": {
    title: "Performance",
    copy: "Bankroll trajectories, replay experiments, and model behavior over time.",
    image: "/images/pages/performance.png",
    signals: ["Trajectory", "Replay", "Drawdown"],
  },
  "/predictions": {
    title: "Model Summary",
    copy: "Pregame model probabilities, market comparison, and feature traceability.",
    image: "/images/pages/predictions.png",
    signals: ["Pregame", "Model layer", "Market lens"],
  },
  "/research-admin": {
    title: "Research Admin",
    copy: "Experiment briefs, run control, and promotion operations for the research loop.",
    image: "/images/pages/research-admin.png",
    signals: ["Briefs", "Runs", "Promotion"],
  },
  "/research-desk": {
    title: "Research Desk",
    copy: "The clean daily view of slate status, evidence posture, and promotion gates.",
    image: "/images/pages/research-desk.png",
    signals: ["Desk status", "Evidence", "Promotion"],
  },
  "/slices": {
    title: "Slice Analysis",
    copy: "Drift and context checks across teams, parks, weather, and game states.",
    image: "/images/pages/slices.png",
    signals: ["Context", "Drift", "Segment"],
  },
  "/validation": {
    title: "Validation",
    copy: "Immutable pregame ledgers, validation gates, and evidence contract checks.",
    image: "/images/pages/validation.png",
    signals: ["Ledger", "Gates", "Audit"],
  },
};

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

function normalizeRoutePath(pathname: string): string {
  const withoutBase =
    BASE_PATH && pathname.startsWith(BASE_PATH) ? pathname.slice(BASE_PATH.length) || "/" : pathname;
  return withoutBase !== "/" ? withoutBase.replace(/\/+$/, "") : withoutBase;
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
  return hrefWithParams(href, searchParams, {
    league,
    strategy: getDefaultBetStrategyForLeague(league),
  });
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
  staticStaging,
  strategy,
  theme,
}: SidebarControlsProps) {
  const isDarkTheme = theme === DARK_THEME;
  const strategyConfig = getBetStrategyConfig(strategy);

  return (
    <>
      <div className="sidebar-brand card">
        <p className="sidebar-eyebrow">Controls</p>
        <h2 className="sidebar-title">Inputs</h2>
        <p className="small sidebar-copy">MLB only. Choose risk, then scan the board.</p>
      </div>

      {SHOW_LEAGUE_SELECTOR ? (
        <section className="sidebar-card card" aria-labelledby="sidebar-league-title">
          <span className="strategy-toggle-label" id="sidebar-league-title">
            League
          </span>
          <div className="league-toggle-row" aria-label="League selection">
            {ALL_LEAGUES.map((code) => (
              <Link
                href={hrefWithLeague(pathname, code, search)}
                key={code}
                className={`league-toggle-btn ${league === code ? "active" : ""}`}
              >
                {displayLeagueLabel(code)}
              </Link>
            ))}
          </div>
        </section>
      ) : (
        <section className="sidebar-card card" aria-labelledby="sidebar-league-title">
          <span className="strategy-toggle-label" id="sidebar-league-title">
            Product Lane
          </span>
          <p className="small">MLB only.</p>
        </section>
      )}

      <section className="sidebar-card card" aria-labelledby="sidebar-profile-title">
        <span className="strategy-toggle-label" id="sidebar-profile-title">
          Bet Objective
        </span>
        <div className="strategy-toggle-row" aria-label="Bet objective selection">
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

      <section className="sidebar-card card" aria-labelledby="sidebar-utilities-title">
        <span className="strategy-toggle-label" id="sidebar-utilities-title">
          Utilities
        </span>
        <div className="refresh-row">
          {!staticStaging ? (
            <button
              type="button"
              className="refresh-btn"
              onClick={onRefresh}
              disabled={isRefreshing}
              aria-busy={isRefreshing}
            >
              {isRefreshing ? (
                <>
                  <span className="refresh-spinner" aria-hidden />
                  Refreshing {displayLeagueLabel(league)}...
                </>
              ) : (
                "Refresh Data"
              )}
            </button>
          ) : null}
          <div className="refresh-meta" aria-live="polite">
            <p className="small">
              {strategyConfig.label}: {strategyConfig.shortLabel}.
            </p>
            {!staticStaging ? <p className="small">Refresh updates ingest only.</p> : null}
            {isRefreshing ? (
              <>
                <p className="small">Refreshing {displayLeagueLabel(league)} data...</p>
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
          league="MLB"
          pathname="/"
          refreshError=""
          refreshedAtLabel=""
          search={new URLSearchParams(`league=MLB&strategy=${getDefaultBetStrategyForLeague("MLB")}`)}
          showRefreshedStamp={false}
          staticStaging={false}
          strategy={getDefaultBetStrategyForLeague("MLB")}
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
  const strategyParam = searchParams.get("strategy");
  const strategy = strategyParam ? normalizeBetStrategy(strategyParam) : getDefaultBetStrategyForLeague(league);
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
          <h1 className="title app-title">Win Probability Forecasting</h1>
          <p className="small dashboard-subtitle">
            Forecasts vs market.
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

function DashboardRouteHero({ pathname }: { pathname: string }) {
  const hero = routeHeroes[normalizeRoutePath(pathname)];
  if (!hero) return null;

  return (
    <section
      className="dashboard-route-hero"
      style={{ "--route-hero-image": `url(${BASE_PATH}${hero.image})` } as CSSProperties}
      aria-labelledby="dashboard-route-hero-title"
    >
      <div className="dashboard-route-hero-content">
        <h2 id="dashboard-route-hero-title">{hero.title}</h2>
        <p>{hero.copy}</p>
      </div>
      <div className="dashboard-route-hero-strip" aria-label={`${hero.title} signals`}>
        {hero.signals.map((signal) => (
          <span key={signal}>{signal}</span>
        ))}
      </div>
    </section>
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
          <h1 className="title app-title">{displayLeagueLabel(league)} Win Probability Forecasting</h1>
          <p className="small dashboard-subtitle">
            Forecasts vs market.
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
      <DashboardRouteHero pathname={pathname} />
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
