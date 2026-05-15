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
  edgeProbability: number;
  edge: string;
  bestBet: string;
  betTone: Tone;
};
export type FrontPageTournamentRow = readonly [string, string, string, string];
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

function hasMarketNumber(value?: number | string | null): boolean {
  if (value === null || value === undefined || value === "") return false;
  return Number.isFinite(Number(value));
}

function americanOddsToProbability(value?: number | null): number | null {
  const price = Number(value);
  if (!Number.isFinite(price) || price === 0) return null;
  return price > 0 ? 100 / (price + 100) : Math.abs(price) / (Math.abs(price) + 100);
}

function moneylineOverlay(row: MarketRow): { edge: number; side: string } {
  const homeModel = Number(row.home_win_probability);
  if (!Number.isFinite(homeModel)) return { edge: 0, side: "No bet" };

  const homeMarketRaw = americanOddsToProbability(row.moneyline?.home_price);
  const awayMarketRaw = americanOddsToProbability(row.moneyline?.away_price);
  if (homeMarketRaw === null || awayMarketRaw === null) return { edge: 0, side: "No bet" };

  const marketTotal = homeMarketRaw + awayMarketRaw;
  if (!Number.isFinite(marketTotal) || marketTotal <= 0) return { edge: 0, side: "No bet" };

  const homeOverlay = homeModel - homeMarketRaw / marketTotal;
  const awayOverlay = (1 - homeModel) - awayMarketRaw / marketTotal;
  if (homeOverlay <= 0 && awayOverlay <= 0) return { edge: 0, side: "No bet" };
  return homeOverlay >= awayOverlay ? { edge: homeOverlay, side: row.home_team } : { edge: awayOverlay, side: row.away_team };
}

function chaosIndexFromOverlay(edge: number): number {
  return Math.max(0, Math.min(100, Math.round(edge * 3000)));
}

function isPricedMoneyline(row: MarketRow): boolean {
  return americanOddsToProbability(row.moneyline?.home_price) !== null && americanOddsToProbability(row.moneyline?.away_price) !== null;
}

function frontPageGame(row: MarketRow): FrontPageGame {
  const overlay = moneylineOverlay(row);
  const hasSpread = hasMarketNumber(row.spread?.point);
  const hasTotal = hasMarketNumber(row.total?.point);

  return {
    time: formatGameTime(row.start_time_utc),
    away: row.away_team,
    home: row.home_team,
    spread: hasSpread ? [`${row.spread.point}`, formatPrice(row.spread.home_price)] : ["Market pending", "No spread"],
    total: hasTotal ? [`${row.total.point}`, formatPrice(row.total.over_price)] : ["Market pending", "No total"],
    chaos: chaosIndexFromOverlay(overlay.edge),
    edgeProbability: overlay.edge,
    edge: formatPercent(overlay.edge, 1),
    bestBet: overlay.side,
    betTone: overlay.edge > 0 ? "teal" : "blue",
  };
}

function tournamentRows(rows?: Array<Record<string, unknown>>): FrontPageTournamentRow[] {
  return (rows || []).slice(0, 3).map((row, index) => [
    String(index + 1),
    String(row.display_name || row.model_name || "Research row"),
    formatPercent(Number(row.roc_auc), 1),
    compactChampionStatus(row.champion_status),
  ]);
}

function compactChampionStatus(value: unknown): string {
  const raw = String(value || "research-only").toLowerCase();
  if (raw.includes("blocked")) return "Blocked";
  if (raw.includes("passed")) return "Passed";
  if (raw.includes("promoted") || raw.includes("approved")) return "Promoted";
  if (raw.includes("challenger")) return "Challenger";
  return "Research";
}

function ensembleRows(): FrontPageEnsembleRow[] {
  return (performance.run_summaries || []).slice(0, 5).map((row) => [
    String(row.model_name || "model"),
    `${Number(row.n_games || 0)} games`,
    Number.isFinite(Number(row.avg_log_loss)) ? Number(row.avg_log_loss).toFixed(3) : "N/A",
    Number.isFinite(Number(row.accuracy)) ? formatPercent(Number(row.accuracy), 1) : "N/A",
  ]);
}

const marketRows = marketBoard.rows || [];
const derivedGames = marketRows.map(frontPageGame);
const pricedSides = marketRows.filter(isPricedMoneyline).length * 2;
const positiveOverlayGames = derivedGames.filter((game) => game.edgeProbability > 0);
const topOverlayGame = positiveOverlayGames.reduce<FrontPageGame | null>(
  (best, game) => (!best || game.edgeProbability > best.edgeProbability ? game : best),
  null,
);
const totalPositiveOverlay = positiveOverlayGames.reduce((sum, game) => sum + game.edgeProbability, 0);

export const frontPageData = {
  modelStamp: formatAsOf(gamesToday.as_of_utc || marketBoard.as_of_utc),
  kpis: [
    { label: "Games Today", value: String(marketRows.length), note: marketBoard.date_central || "MLB snapshot", noteTone: "teal" },
    {
      label: "Top Edge",
      value: topOverlayGame ? topOverlayGame.edge : "Pending",
      note: topOverlayGame ? `${topOverlayGame.away} @ ${topOverlayGame.home}` : "No sportsbook odds",
      noteTone: "orange",
    },
    {
      label: "Best Bet",
      value: topOverlayGame?.bestBet || "No bet",
      note: topOverlayGame ? `Edge ${topOverlayGame.edge}` : "Awaiting prices",
      noteTone: "teal",
    },
    { label: "Positive EV", value: String(positiveOverlayGames.length), note: `of ${pricedSides} priced sides`, noteTone: "teal" },
    { label: "Total Edge", value: formatPercent(totalPositiveOverlay, 2), note: "market overlays", noteTone: "teal" },
  ] as FrontPageKpi[],
  games: derivedGames.slice(0, 8),
  upcomingStarters: marketRows.slice(8, 11).map((row) => ({
    time: formatGameTime(row.start_time_utc),
    away: row.away_team,
    home: row.home_team,
  })),
  intraFamilies: tournamentRows(nestedTournament.family_champions),
  interFamilies: tournamentRows(nestedTournament.inter_family_leaderboard),
  ensembleRows: ensembleRows(),
};
