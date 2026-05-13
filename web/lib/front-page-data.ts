import gamesTodaySnapshot from "@/public/staging-data/mlb/games-today.json";
import marketBoardSnapshot from "@/public/staging-data/mlb/market-board.json";
import nestedTournamentSnapshot from "@/public/staging-data/mlb/nested-tournament.json";
import performanceSnapshot from "@/public/staging-data/mlb/performance.json";
import { normalizeUtcTimestamp } from "@/lib/games-today";
import type { GamesTodayResponse, MarketBoardResponse, NestedTournamentResponse } from "@/lib/types";

type Tone = "blue" | "orange" | "red" | "teal";
type MarketRow = NonNullable<MarketBoardResponse["rows"]>[number];

export type FrontPageKpi = { label: string; value: string; note: string; noteTone: "orange" | "teal" };
export type FrontPageGame = {
  time: string;
  away: string;
  home: string;
  spread: readonly [string, string];
  total: readonly [string, string];
  chaos: number;
  edge: string;
  bestBet: string;
  betTone: Tone;
};
export type FrontPageTournamentRow = readonly [string, string, string, string, string];
export type FrontPageEnsembleRow = readonly [string, string, string, string];

const gamesToday = gamesTodaySnapshot as unknown as GamesTodayResponse;
const marketBoard = marketBoardSnapshot as unknown as MarketBoardResponse;
const nestedTournament = nestedTournamentSnapshot as NestedTournamentResponse;
const performance = performanceSnapshot as {
  run_summaries?: Array<{ model_name?: string; n_games?: number; avg_log_loss?: number; avg_brier?: number; accuracy?: number }>;
};

function formatAsOf(value?: string | null): string {
  if (!value) return "Snapshot unavailable";
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return value;
  return `${parsed.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "America/Chicago" })} ${parsed.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/Chicago" })} CT`;
}

function formatGameTime(value?: string | null): string {
  if (!value) return "TBD";
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return "TBD";
  return parsed.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York" });
}

function formatPrice(value?: number | null): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric === 0) return "No odds";
  const rounded = Math.round(numeric);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

function formatPercent(value?: number | null, digits = 1): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(digits)}%` : "N/A";
}

function modelDisagreement(row: MarketRow): number {
  const values = Object.values(row.model_win_probabilities || {}).filter((value): value is number => Number.isFinite(value));
  if (values.length < 2) return 0;
  return Math.max(...values) - Math.min(...values);
}

function frontPageGame(row: MarketRow): FrontPageGame {
  const homeProb = Number(row.home_win_probability);
  const favorite = Number.isFinite(homeProb) && homeProb >= 0.5 ? row.home_team : row.away_team;
  const favoriteProb = Number.isFinite(homeProb) ? Math.max(homeProb, 1 - homeProb) : null;
  const disagreement = modelDisagreement(row);
  const hasSpread = Number.isFinite(Number(row.spread?.point));
  const hasTotal = Number.isFinite(Number(row.total?.point));

  return {
    time: formatGameTime(row.start_time_utc),
    away: row.away_team,
    home: row.home_team,
    spread: hasSpread ? [`${row.spread.point}`, formatPrice(row.spread.home_price)] : ["Market pending", "No spread"],
    total: hasTotal ? [`${row.total.point}`, formatPrice(row.total.over_price)] : ["Market pending", "No total"],
    chaos: Math.max(1, Math.min(100, Math.round(disagreement * 500))),
    edge: favoriteProb === null ? "N/A" : formatPercent(favoriteProb - 0.5, 1),
    bestBet: row.moneyline?.books_count ? favorite : "No bet",
    betTone: row.moneyline?.books_count ? "teal" : "blue",
  };
}

function tournamentRows(rows?: Array<Record<string, unknown>>): FrontPageTournamentRow[] {
  return (rows || []).slice(0, 3).map((row, index) => [
    String(index + 1),
    String(row.display_name || row.model_name || "Research row"),
    formatPercent(Number(row.roc_auc), 1),
    String(row.champion_status || "research-only"),
    formatPercent(Number(row.accuracy), 1),
  ]);
}

function ensembleRows(): FrontPageEnsembleRow[] {
  return (performance.run_summaries || []).slice(0, 5).map((row) => [
    String(row.model_name || "model"),
    `${Number(row.n_games || 0)} games`,
    Number.isFinite(Number(row.avg_log_loss)) ? Number(row.avg_log_loss).toFixed(3) : "N/A",
    Number.isFinite(Number(row.accuracy)) ? formatPercent(Number(row.accuracy), 1) : "N/A",
  ]);
}

export const frontPageData = {
  modelStamp: formatAsOf(gamesToday.as_of_utc || marketBoard.as_of_utc),
  kpis: [
    { label: "Games Today", value: String(marketBoard.rows?.length || 0), note: marketBoard.date_central || "MLB snapshot", noteTone: "teal" },
    { label: "Top Signal", value: "Market pending", note: "No sportsbook odds", noteTone: "orange" },
    { label: "Best Bet", value: "No bet", note: "Awaiting prices", noteTone: "teal" },
    { label: "Positive EV", value: "0", note: "contract-safe fallback", noteTone: "teal" },
    { label: "Model Edge", value: "Prob. lean", note: "vs 50/50 baseline", noteTone: "teal" },
  ] as FrontPageKpi[],
  games: (marketBoard.rows || []).slice(0, 8).map(frontPageGame),
  upcomingStarters: (marketBoard.rows || []).slice(8, 11).map((row) => ({
    time: formatGameTime(row.start_time_utc),
    away: row.away_team,
    home: row.home_team,
  })),
  intraFamilies: tournamentRows(nestedTournament.family_champions),
  interFamilies: tournamentRows(nestedTournament.inter_family_leaderboard),
  ensembleRows: ensembleRows(),
};
