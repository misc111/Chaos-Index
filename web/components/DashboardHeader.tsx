"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { getDefaultBetStrategyForLeague } from "@/lib/betting-strategy";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const DEFAULT_QUERY = `?league=MLB&strategy=${getDefaultBetStrategyForLeague("MLB")}`;

const links: Array<[string, string]> = [
  ["/", "Front page"],
  ["/games-today", "Games today"],
  ["/intra-family-tournament", "Intra-Family Tournament"],
  ["/inter-family-tournament", "Inter-Family Tournament"],
  ["/ensemble-summary", "Ensemble Summary Table"],
];

function normalizePath(pathname: string): string {
  const withoutBase = BASE_PATH && pathname.startsWith(BASE_PATH) ? pathname.slice(BASE_PATH.length) || "/" : pathname;
  return withoutBase !== "/" ? withoutBase.replace(/\/+$/, "") : withoutBase;
}

function isActivePath(pathname: string, href: string): boolean {
  return normalizePath(pathname) === href;
}

function toBrowserHref(href: string, query: string): string {
  const withQuery = href === "/" ? href : `${href}${query}`;
  if (!BASE_PATH || !withQuery.startsWith("/")) return withQuery;
  return withQuery === "/" ? `${BASE_PATH}/` : `${BASE_PATH}${withQuery}`;
}

function HeaderContent() {
  const pathname = usePathname() || "/";
  const searchParams = useSearchParams();
  const query = searchParams.toString() ? `?${searchParams.toString()}` : DEFAULT_QUERY;
  const isHome = isActivePath(pathname, "/");

  return (
    <header className={`site-header ${isHome ? "site-header-home" : ""}`}>
      <Link href={toBrowserHref("/", "")} className="site-brand" aria-label="Chaos Index front page">
        <span className="site-brand-mark" aria-hidden="true">
          CI
        </span>
        <span className="site-brand-copy">
          <span>Chaos Index</span>
          <small>baseball probabilities</small>
        </span>
      </Link>
      {!isHome ? (
        <nav className="site-nav" aria-label="Primary navigation">
          {links.map(([href, label]) => {
            const isActive = isActivePath(pathname, href);
            return (
              <Link
                href={toBrowserHref(href, query)}
                key={href}
                className={`site-nav-link ${isActive ? "active" : ""}`}
                aria-current={isActive ? "page" : undefined}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      ) : null}
    </header>
  );
}

function HeaderFallback() {
  return (
    <header className="site-header">
      <a href={toBrowserHref("/", "")} className="site-brand">
        <span className="site-brand-mark" aria-hidden="true">
          CI
        </span>
        <span className="site-brand-copy">
          <span>Chaos Index</span>
          <small>baseball probabilities</small>
        </span>
      </a>
      <nav className="site-nav" aria-label="Primary navigation">
        {links.map(([href, label]) => (
          <a href={toBrowserHref(href, DEFAULT_QUERY)} className="site-nav-link" key={href}>
            {label}
          </a>
        ))}
      </nav>
    </header>
  );
}

export function DashboardSidebar() {
  return null;
}

export default function DashboardHeader() {
  return (
    <Suspense fallback={<HeaderFallback />}>
      <HeaderContent />
    </Suspense>
  );
}
