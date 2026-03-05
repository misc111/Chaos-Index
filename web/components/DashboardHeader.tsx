"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { type LeagueCode, normalizeLeague } from "@/lib/league";

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

function hrefWithLeague(href: string, league: LeagueCode, searchParams: URLSearchParams): string {
  const params = new URLSearchParams(searchParams.toString());
  params.set("league", league);
  const query = params.toString();
  return query ? `${href}?${query}` : href;
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
