import { NextResponse } from "next/server.js";
import { computeBetDecisionsForSlate, type BetDecision } from "@/lib/betting";
import { getBetStrategyConfig, strategyFromRequest } from "@/lib/betting-strategy";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest, type LeagueCode } from "@/lib/league";
import { getActiveBetRiskRegime, getActiveChampionSummary } from "@/lib/server/services/betting-driver";
import { getMarketBoardPayload } from "@/lib/server/services/market-board";
import type {
  ResearchChampionSummary,
  ResearchDeskNightlyRow,
  ResearchDeskResponse,
  ResearchPromotionSummary,
  TableRow,
} from "@/lib/types";

export const dynamic = "force-dynamic";

type NightlyBoardInput = {
  rows: Array<{
    game_id: number;
    start_time_utc?: string | null;
    home_team_name: string;
    away_team_name: string;
    home_win_probability: number;
    betting_model_name?: string | null;
  }>;
};

type RawPromotionRow = {
  promoted?: number | null;
  incumbent_model_name?: string | null;
  candidate_model_name?: string | null;
  reason_summary?: string | null;
  policy_json?: string | null;
  created_at_utc?: string | null;
};

const NBA_ONLY_SUMMARY =
  "Research desk is piloting on NBA only in v1. Other leagues keep their existing dashboard surfaces while promotion and nightly underwriting stay focused on NBA.";

function parseJsonRecord(value?: string | null): TableRow | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function normalizePromotionReason(value?: string | null): string | null {
  const raw = String(value || "").trim();
  if (!raw) return null;
  return raw.replace(/^Rejected:\s*/i, "").replace(/^Auto-promoted\s*/i, "Auto-promoted");
}

export function buildUnsupportedPayload(league: LeagueCode): ResearchDeskResponse {
  return {
    league,
    as_of_utc: null,
    odds_as_of_utc: null,
    date_central: undefined,
    desk_posture: "normal",
    overnight_summary: NBA_ONLY_SUMMARY,
    champion: null,
    latest_promotion: null,
    counts: {
      total_games: 0,
      bets: 0,
      passes: 0,
    },
    rows: [],
  };
}

function loadLatestPromotion(league: LeagueCode): ResearchPromotionSummary | null {
  try {
    const rows = runSqlJson<RawPromotionRow>(
      `SELECT
         promoted,
         incumbent_model_name,
         candidate_model_name,
         reason_summary,
         policy_json,
         created_at_utc
       FROM promotion_decisions
       WHERE league = ${JSON.stringify(league)}
         AND profile_key = 'default'
       ORDER BY created_at_utc DESC, decision_id DESC
       LIMIT 1`,
      { league }
    );
    const row = rows[0];
    if (!row) return null;
    return {
      promoted: Boolean(Number(row.promoted || 0)),
      incumbent_model_name: row.incumbent_model_name ? String(row.incumbent_model_name) : null,
      candidate_model_name: row.candidate_model_name ? String(row.candidate_model_name) : null,
      reason_summary: row.reason_summary ? String(row.reason_summary) : null,
      policy: parseJsonRecord(row.policy_json),
      created_at_utc: row.created_at_utc ? String(row.created_at_utc) : null,
    };
  } catch {
    return null;
  }
}

function loadChampionSummary(league: LeagueCode): ResearchChampionSummary | null {
  const champion = getActiveChampionSummary(league);
  if (!champion?.model_name || !champion.promoted_at_utc) {
    return null;
  }
  return {
    league: champion.league,
    profile_key: champion.profile_key,
    model_name: champion.model_name,
    promoted_at_utc: champion.promoted_at_utc,
    source_run_id: champion.source_run_id ?? null,
    source_brief_id: champion.source_brief_id ?? null,
    descriptor: champion.descriptor ?? null,
    policy: champion.policy ?? null,
  };
}

export function buildNightlyRows(
  marketBoard: NightlyBoardInput,
  decisions: BetDecision[]
): ResearchDeskNightlyRow[] {
  return marketBoard.rows.map((row, index) => {
    const decision = decisions[index];
    const isBet = Boolean(decision && decision.stake > 0 && decision.side !== "none");
    return {
      game_id: row.game_id,
      start_time_utc: row.start_time_utc,
      home_team: row.home_team_name,
      away_team: row.away_team_name,
      bet_label: isBet ? "bet" : "pass",
      reason: decision?.reason || "No recommendation available",
      side: decision?.side || "none",
      team: decision?.team || null,
      stake: decision?.stake || 0,
      odds: decision?.odds ?? null,
      edge: decision?.edge ?? null,
      expected_value: decision?.expectedValue ?? null,
      home_win_probability: row.home_win_probability,
      betting_model_name: row.betting_model_name ?? null,
    };
  });
}

export function buildOvernightSummary(args: {
  deskPosture: "normal" | "guarded";
  championModelName?: string | null;
  promotion: ResearchPromotionSummary | null;
  totalGames: number;
  betCount: number;
}): string {
  const postureSentence =
    args.deskPosture === "guarded"
      ? "Guarded posture is active, so the desk is using tighter NBA risk controls tonight."
      : "Normal posture is active, so the desk is running on its standard NBA underwriting rules.";
  const championSentence = args.championModelName
    ? `Active champion: ${args.championModelName}.`
    : "No promoted champion is recorded yet, so the desk is leaning on the current fallback model.";
  const promotionSentence = args.promotion
    ? args.promotion.promoted
      ? `Latest promotion: ${args.promotion.candidate_model_name || "candidate"} cleared the gates.`
      : `Latest promotion review stayed put: ${normalizePromotionReason(args.promotion.reason_summary) || "the last candidate did not clear the gates"}.`
    : "No promotion decision has been recorded yet.";
  const slateSentence =
    args.totalGames > 0
      ? `Tonight's slate has ${pluralize(args.totalGames, "game")}, with ${pluralize(args.betCount, "bet")} and ${pluralize(
          Math.max(0, args.totalGames - args.betCount),
          "pass"
        )}.`
      : "No NBA games are on the desk slate right now.";

  return [postureSentence, championSentence, promotionSentence, slateSentence].join(" ");
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  if (league !== "NBA") {
    return NextResponse.json(buildUnsupportedPayload(league));
  }

  const marketBoard = await getMarketBoardPayload(league);
  const riskRegime = getActiveBetRiskRegime(league);
  const strategy = strategyFromRequest(request);
  const strategyConfig = marketBoard.strategy_configs?.[strategy] || getBetStrategyConfig(strategy, { league, riskRegime });
  const decisions = computeBetDecisionsForSlate(
    marketBoard.rows.map((row) => ({
      league,
      home_team: row.home_team_name,
      away_team: row.away_team_name,
      home_win_probability: row.home_win_probability,
      home_moneyline: row.moneyline.home_price,
      away_moneyline: row.moneyline.away_price,
      betting_model_name: row.betting_model_name,
      model_win_probabilities: row.model_win_probabilities,
    })),
    strategy,
    strategyConfig,
    undefined,
    { league, riskRegime }
  );

  const rows = buildNightlyRows(marketBoard, decisions);
  const champion = loadChampionSummary(league);
  const latestPromotion = loadLatestPromotion(league);
  const counts = {
    total_games: rows.length,
    bets: rows.filter((row) => row.bet_label === "bet").length,
    passes: rows.filter((row) => row.bet_label === "pass").length,
  };

  return NextResponse.json({
    league,
    as_of_utc: marketBoard.as_of_utc,
    odds_as_of_utc: marketBoard.odds_as_of_utc,
    date_central: marketBoard.date_central,
    desk_posture: riskRegime,
    overnight_summary: buildOvernightSummary({
      deskPosture: riskRegime,
      championModelName: champion?.model_name,
      promotion: latestPromotion,
      totalGames: counts.total_games,
      betCount: counts.bets,
    }),
    champion,
    latest_promotion: latestPromotion,
    counts,
    rows,
  } satisfies ResearchDeskResponse);
}
