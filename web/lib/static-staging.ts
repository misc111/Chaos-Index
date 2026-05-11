import { type LeagueCode } from "@/lib/league";
import { buildPerformanceExperimentStagingFileName } from "@/lib/performance-replay-experiments";

// Maintainer note: GitHub Pages staging does not call the live API routes.
// It reads committed JSON snapshots from web/public/staging-data/, so any
// dashboard/API change that should show up on staging also needs a fresh
// `npm run generate:staging-data` and committed snapshot files.
const STATIC_STAGING = process.env.NEXT_PUBLIC_STATIC_STAGING === "1";
const PUBLIC_VIEWS_USE_STATIC_SNAPSHOTS = true;
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const STAGING_ASSET_VERSION = process.env.NEXT_PUBLIC_STAGING_ASSET_VERSION || "";

const STAGING_FILES = {
  actualVsExpected: "actual-vs-expected.json",
  betHistory: "bet-history.json",
  gamesToday: "games-today.json",
  marketBoard: "market-board.json",
  metrics: "metrics.json",
  currentSeasonPredictiveness: "current-season-predictiveness.json",
  nestedTournament: "nested-tournament.json",
  performance: "performance.json",
  predictions: "predictions.json",
  validation: "validation.json",
} as const;

export type StagingDataKey = keyof typeof STAGING_FILES;

function resolveStagingFileName(key: StagingDataKey, stagingVariant?: string | null): string {
  if (key === "performance") {
    return buildPerformanceExperimentStagingFileName(stagingVariant);
  }
  return STAGING_FILES[key];
}

export function isStaticStagingBuild(): boolean {
  return STATIC_STAGING;
}

export function buildStaticStagingUrl(key: StagingDataKey, league: LeagueCode, stagingVariant?: string | null): string {
  const path = `${BASE_PATH}/staging-data/${league.toLowerCase()}/${resolveStagingFileName(key, stagingVariant)}`;
  return STAGING_ASSET_VERSION ? `${path}?v=${encodeURIComponent(STAGING_ASSET_VERSION)}` : path;
}

export function buildDashboardDataUrl(
  key: StagingDataKey,
  livePath: string,
  league: LeagueCode,
  stagingVariant?: string | null
): string {
  void livePath;
  return buildStaticStagingUrl(key, league, stagingVariant);
}

export async function fetchDashboardJson<T>(
  key: StagingDataKey,
  livePath: string,
  league: LeagueCode,
  stagingVariant?: string | null,
  signal?: AbortSignal
): Promise<T> {
  const response = await fetch(buildDashboardDataUrl(key, livePath, league, stagingVariant), {
    cache: STATIC_STAGING || PUBLIC_VIEWS_USE_STATIC_SNAPSHOTS ? "force-cache" : "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(`Request failed (${response.status}).`);
  }

  return (await response.json()) as T;
}
