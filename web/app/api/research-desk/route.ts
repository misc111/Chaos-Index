import { NextResponse } from "next/server.js";
import fs from "node:fs";
import path from "node:path";
import { computeBetDecisionsForSlate, type BetDecision } from "@/lib/betting";
import { getBetStrategyConfig, strategyFromRequest } from "@/lib/betting-strategy";
import { runSqlJson } from "@/lib/db";
import { displayLeagueLabel, leagueFromRequest, type LeagueCode } from "@/lib/league";
import { predictionTrustNote } from "@/lib/predictions-report";
import { loadServerModelFeatureMap } from "@/lib/server/repositories/model-feature-map";
import { getActiveBetRiskRegime, getActiveChampionSummary } from "@/lib/server/services/betting-driver";
import { getMarketBoardPayload } from "@/lib/server/services/market-board";
import { buildPredictionModelSummary } from "@/lib/server/services/predictions";
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

function parseJsonRecord(value?: string | null): TableRow | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function readJsonRecord(filePath: string): TableRow | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function loadEvidenceStatus(league: LeagueCode): Pick<
  ResearchDeskResponse,
  "source_kind" | "source_status" | "evidence_stage" | "latest_artifact_role" | "promotion_eligible" | "production_ready" | "evidence_status"
> {
  if (league !== "MLB") {
    return {
      source_kind: null,
      source_status: null,
      evidence_stage: null,
      latest_artifact_role: null,
      promotion_eligible: false,
      production_ready: false,
      evidence_status: null,
    };
  }
  const repoRoot = path.resolve(process.cwd(), "..");
  const currentBest = readJsonRecord(path.resolve(repoRoot, "artifacts", "reports", league.toLowerCase(), "current_best_models.json"));
  const evidenceStatus = parseJsonRecord(JSON.stringify(currentBest?.evidence_status ?? null));
  const latestArtifactRole =
    typeof currentBest?.latest_artifact_role === "string" ? String(currentBest.latest_artifact_role) : null;
  const evidenceStage =
    typeof evidenceStatus?.evidence_stage === "string" ? String(evidenceStatus.evidence_stage) : null;
  return {
    source_kind: typeof currentBest?.source_kind === "string" ? String(currentBest.source_kind) : null,
    source_status: typeof currentBest?.source_status === "string" ? String(currentBest.source_status) : null,
    evidence_stage: evidenceStage,
    latest_artifact_role: latestArtifactRole,
    promotion_eligible: Boolean(currentBest?.promotion_eligible === true),
    production_ready: Boolean(currentBest?.production_ready === true || evidenceStatus?.production_ready === true),
    evidence_status: evidenceStatus,
  };
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function displayModelName(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) return "candidate";
  return raw
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((token) => {
      const lower = token.toLowerCase();
      if (["glm", "gam", "dglm", "mars"].includes(lower)) return lower.toUpperCase();
      return token.charAt(0).toUpperCase() + token.slice(1);
    })
    .join(" ");
}

function summarizePromotion(promotion: ResearchPromotionSummary | null): string {
  if (!promotion) return "No promotion review yet.";
  const candidate = displayModelName(promotion.candidate_model_name);
  return promotion.promoted ? `${candidate} promoted.` : `${candidate} stayed in research.`;
}

export function buildUnsupportedPayload(league: LeagueCode): ResearchDeskResponse {
  const emptySummary = buildOvernightSummary({
    league,
    deskPosture: "normal",
    championModelName: null,
    promotion: null,
    totalGames: 0,
    betCount: 0,
  });

  return {
    league,
    as_of_utc: null,
    odds_as_of_utc: null,
    date_central: undefined,
    desk_posture: "normal",
    overnight_summary: emptySummary,
    champion: null,
    model_diagnostics: {},
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
  league: LeagueCode;
  deskPosture: "normal" | "guarded";
  championModelName?: string | null;
  promotion: ResearchPromotionSummary | null;
  totalGames: number;
  betCount: number;
}): string {
  const leagueLabel = displayLeagueLabel(args.league);
  const riskLabel = args.deskPosture === "guarded" ? "Guarded risk" : "Normal risk";
  const modelSentence = args.championModelName
    ? `Champion: ${displayModelName(args.championModelName)}.`
    : "Fallback model active.";
  const slateLead =
    args.totalGames > 0
      ? `${leagueLabel} desk: ${pluralize(args.totalGames, "game")}, ${pluralize(args.betCount, "bet")}, ${pluralize(
          Math.max(0, args.totalGames - args.betCount),
          "pass",
          "passes"
        )}.`
      : `${leagueLabel} desk: no games loaded.`;

  return [slateLead, riskLabel + ".", modelSentence, summarizePromotion(args.promotion)].join(" ");
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  try {
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
    const featureMap = await loadServerModelFeatureMap(league);
    const modelDiagnostics = Object.fromEntries(
      Object.entries(featureMap.models).map(([modelName, diagnostics]) => [
        modelName,
        buildPredictionModelSummary(modelName, league, predictionTrustNote(modelName, league), diagnostics),
      ])
    );
    const latestPromotion = loadLatestPromotion(league);
    const evidenceStatus = loadEvidenceStatus(league);
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
        league,
        deskPosture: riskRegime,
        championModelName: champion?.model_name,
        promotion: latestPromotion,
        totalGames: counts.total_games,
        betCount: counts.bets,
      }),
      champion,
      model_diagnostics: modelDiagnostics,
      model_feature_map_updated_at_utc: featureMap.updated_at_utc,
      model_feature_map_source: featureMap.source,
      model_feature_map_run_id: featureMap.model_run_id,
      model_feature_set_version: featureMap.feature_set_version,
      latest_promotion: latestPromotion,
      source_kind: evidenceStatus.source_kind,
      source_status: evidenceStatus.source_status,
      evidence_stage: evidenceStatus.evidence_stage,
      latest_artifact_role: evidenceStatus.latest_artifact_role,
      promotion_eligible: evidenceStatus.promotion_eligible,
      production_ready: evidenceStatus.production_ready,
      evidence_status: evidenceStatus.evidence_status,
      counts,
      rows,
    } satisfies ResearchDeskResponse);
  } catch {
    return NextResponse.json(buildUnsupportedPayload(league));
  }
}
