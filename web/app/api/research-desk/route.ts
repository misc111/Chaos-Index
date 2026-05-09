import { NextResponse } from "next/server.js";
import { computeBetDecisionsForSlate } from "@/lib/betting";
import { getBetStrategyConfig, strategyFromRequest } from "@/lib/betting-strategy";
import { leagueFromRequest } from "@/lib/league";
import { predictionTrustNote } from "@/lib/predictions-report";
import { loadServerModelFeatureMap } from "@/lib/server/repositories/model-feature-map";
import { getActiveBetRiskRegime } from "@/lib/server/services/betting-driver";
import { getMarketBoardPayload } from "@/lib/server/services/market-board";
import { buildPredictionModelSummary } from "@/lib/server/services/predictions";
import type { ResearchDeskResponse } from "@/lib/types";
import {
  buildNightlyRows,
  buildOvernightSummary,
  buildUnsupportedPayload,
  loadChampionSummary,
  loadEvidenceStatus,
  loadLatestPromotion,
} from "./route-support";

export const dynamic = "force-dynamic";

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
      current_best_top_models: evidenceStatus.current_best_top_models,
      counts,
      rows,
    } satisfies ResearchDeskResponse);
  } catch {
    return NextResponse.json(buildUnsupportedPayload(league));
  }
}
