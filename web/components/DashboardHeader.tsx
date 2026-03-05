"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { type LeagueCode, normalizeLeague, withLeague } from "@/lib/league";

const links: Array<[string, string]> = [
  ["/", "Overview"],
  ["/predictions", "Predictions"],
  ["/actual-vs-expected", "Actual vs Expected"],
  ["/leaderboard", "Leaderboard"],
  ["/performance", "Performance"],
  ["/calibration", "Calibration"],
  ["/diagnostics", "Diagnostics"],
  ["/slices", "Slices"],
  ["/validation", "Validation"],
];

type RefreshResponse = {
  ok?: boolean;
  error?: string;
  details?: string;
  refreshed_at_utc?: string;
};

function hrefWithLeague(href: string, league: LeagueCode, searchParams: URLSearchParams): string {
  const params = new URLSearchParams(searchParams.toString());
  params.set("league", league);
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
      <h1 className="title app-title">NHL Win Probability Forecasting</h1>
      <div className="league-toggle-row" aria-label="League selection">
        <Link href="?league=NHL" className="league-toggle-btn active">
          NHL
        </Link>
        <Link href="?league=NBA" className="league-toggle-btn">
          NBA
        </Link>
      </div>
      <div className="refresh-row">
        <button type="button" className="refresh-btn" disabled>
          Refresh Data
        </button>
      </div>
      <div className="nav">
        {links.map(([href, label]) => (
          <Link href={`${href}?league=NHL`} key={href}>
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
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState("");
  const refreshedAtRaw = searchParams.get("refreshedAt");
  const refreshedLeague = searchParams.get("refreshedLeague");

  const showRefreshedStamp =
    typeof refreshedAtRaw === "string" && refreshedAtRaw.length > 0 && refreshedLeague === league;

  const refreshedAtLabel = showRefreshedStamp ? formatRefreshTimestamp(refreshedAtRaw) : "";

  const handleRefresh = async () => {
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

  return (
    <>
      <h1 className="title app-title">{league} Win Probability Forecasting</h1>

      <div className="league-toggle-row" aria-label="League selection">
        {(["NHL", "NBA"] as LeagueCode[]).map((code) => (
          <Link
            href={hrefWithLeague(pathname, code, search)}
            key={code}
            className={`league-toggle-btn ${league === code ? "active" : ""}`}
          >
            {code}
          </Link>
        ))}
      </div>

      <div className="refresh-row">
        <button
          type="button"
          className="refresh-btn"
          onClick={handleRefresh}
          disabled={isRefreshing}
          aria-busy={isRefreshing}
        >
          {isRefreshing ? (
            <>
              <span className="refresh-spinner" aria-hidden />
              Refreshing {league}...
            </>
          ) : (
            "Refresh Data"
          )}
        </button>
        <div className="refresh-meta" aria-live="polite">
          {isRefreshing ? (
            <>
              <p className="small">Refreshing {league} data...</p>
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
        {links.map(([href, label]) => (
          <Link href={hrefWithLeague(href, league, search)} key={href}>
            {label}
          </Link>
        ))}
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
