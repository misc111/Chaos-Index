import { type LeagueCode, withLeague } from "@/lib/league";

const STATIC_STAGING = process.env.NEXT_PUBLIC_STATIC_STAGING === "1";
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const STAGING_ASSET_VERSION = process.env.NEXT_PUBLIC_STAGING_ASSET_VERSION || "";

const STAGING_FILES = {
  actualVsExpected: "actual-vs-expected.json",
  betHistory: "bet-history.json",
  gamesToday: "games-today.json",
  marketBoard: "market-board.json",
  metrics: "metrics.json",
  performance: "performance.json",
  predictions: "predictions.json",
  validation: "validation.json",
} as const;

export type StagingDataKey = keyof typeof STAGING_FILES;

export function isStaticStagingBuild(): boolean {
  return STATIC_STAGING;
}

export function buildStaticStagingUrl(key: StagingDataKey, league: LeagueCode): string {
  const path = `${BASE_PATH}/staging-data/${league.toLowerCase()}/${STAGING_FILES[key]}`;
  return STAGING_ASSET_VERSION ? `${path}?v=${encodeURIComponent(STAGING_ASSET_VERSION)}` : path;
}

export function buildDashboardDataUrl(key: StagingDataKey, livePath: string, league: LeagueCode): string {
  return STATIC_STAGING ? buildStaticStagingUrl(key, league) : withLeague(livePath, league);
}

export async function fetchDashboardJson<T>(key: StagingDataKey, livePath: string, league: LeagueCode): Promise<T> {
  const response = await fetch(buildDashboardDataUrl(key, livePath, league), {
    cache: STATIC_STAGING ? "force-cache" : "no-store",
  });

  if (!response.ok) {
    throw new Error(`Request failed (${response.status}).`);
  }

  return (await response.json()) as T;
}
