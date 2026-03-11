import { computeBetDecisionsForSlate, settleBet } from "@/lib/betting";
import { BET_STRATEGIES, getBetStrategyConfig } from "@/lib/betting-strategy";
import type { ModelWinProbabilities } from "@/lib/betting-model";
import type {
  ModelReplayBetRow,
  ModelReplayDecisionDetail,
  ModelReplayRunRow,
  ModelReplayStrategySummary,
  ModelRunSummaryRow,
  TableRow,
} from "@/lib/types";

type ModelReplayStrategyKey = keyof ModelReplayRunRow["strategies"];

export const MODEL_REPLAY_COMPARISON_STRATEGIES: readonly ModelReplayStrategyKey[] = BET_STRATEGIES;

export type ModelReplayCandidateRow = {
  game_id: number;
  date_central: string;
  start_time_utc: string | null;
  final_utc: string | null;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_win: number | null;
  forecast_as_of_utc: string;
  home_moneyline: number;
  away_moneyline: number;
  model_name: string;
  model_run_id: string;
  home_win_probability: number;
  model_win_probabilities: ModelWinProbabilities;
};

export type ModelReplayRunMetadata = {
  model_name: string;
  model_run_id: string;
  run_type?: string | null;
  created_at_utc?: string | null;
  snapshot_id?: string | null;
  artifact_path?: string | null;
  feature_set_version?: string | null;
  feature_columns?: string[];
  feature_metadata?: TableRow | null;
  params?: TableRow | null;
  metrics?: TableRow | null;
};

type MutableStrategySummary = {
  total_games: number;
  suggested_bets: number;
  wins: number;
  losses: number;
  total_risked: number;
  total_profit: number;
  first_bet_date_central: string | null;
  last_bet_date_central: string | null;
  edge_sum: number;
  edge_count: number;
  expected_value_sum: number;
  expected_value_count: number;
};

type MutableReplayBetRow = Omit<ModelReplayBetRow, "strategies"> & {
  strategies: Partial<ModelReplayBetRow["strategies"]>;
};

function emptyStrategySummary(totalGames: number): MutableStrategySummary {
  return {
    total_games: totalGames,
    suggested_bets: 0,
    wins: 0,
    losses: 0,
    total_risked: 0,
    total_profit: 0,
    first_bet_date_central: null,
    last_bet_date_central: null,
    edge_sum: 0,
    edge_count: 0,
    expected_value_sum: 0,
    expected_value_count: 0,
  };
}

function finalizeStrategySummary(summary: MutableStrategySummary): ModelReplayStrategySummary {
  return {
    total_games: summary.total_games,
    suggested_bets: summary.suggested_bets,
    wins: summary.wins,
    losses: summary.losses,
    total_risked: summary.total_risked,
    total_profit: summary.total_profit,
    roi: summary.total_risked > 0 ? summary.total_profit / summary.total_risked : 0,
    avg_edge: summary.edge_count > 0 ? summary.edge_sum / summary.edge_count : null,
    avg_expected_value: summary.expected_value_count > 0 ? summary.expected_value_sum / summary.expected_value_count : null,
    first_bet_date_central: summary.first_bet_date_central,
    last_bet_date_central: summary.last_bet_date_central,
  };
}

function buildDecisionDetail(
  row: ModelReplayCandidateRow,
  decision: ReturnType<typeof computeBetDecisionsForSlate>[number]
): ModelReplayDecisionDetail {
  const settlement = settleBet(decision, row.home_win);
  return {
    bet_label: decision.bet,
    reason: decision.reason,
    side: decision.side,
    team: decision.team,
    stake: decision.stake,
    odds: decision.odds,
    model_probability: decision.modelProbability,
    market_probability: decision.marketProbability,
    edge: decision.edge,
    expected_value: decision.expectedValue,
    outcome: settlement.outcome,
    profit: settlement.profit,
    payout: settlement.payout,
  };
}

function trackStrategyOutcome(summary: MutableStrategySummary, detail: ModelReplayDecisionDetail, dateCentral: string): void {
  if (detail.stake <= 0 || detail.outcome === "no_bet") {
    return;
  }

  summary.suggested_bets += 1;
  summary.total_risked += detail.stake;
  summary.total_profit += detail.profit;
  if (detail.outcome === "win") summary.wins += 1;
  if (detail.outcome === "loss") summary.losses += 1;
  if (typeof detail.edge === "number" && Number.isFinite(detail.edge)) {
    summary.edge_sum += detail.edge;
    summary.edge_count += 1;
  }
  if (typeof detail.expected_value === "number" && Number.isFinite(detail.expected_value)) {
    summary.expected_value_sum += detail.expected_value;
    summary.expected_value_count += 1;
  }
  if (!summary.first_bet_date_central || dateCentral < summary.first_bet_date_central) {
    summary.first_bet_date_central = dateCentral;
  }
  if (!summary.last_bet_date_central || dateCentral > summary.last_bet_date_central) {
    summary.last_bet_date_central = dateCentral;
  }
}

function sortTimestamp(createdAt?: string | null, fallbackDate?: string | null): number {
  const fallback = fallbackDate ? `${fallbackDate}T23:59:59Z` : "";
  const value = Date.parse(String(createdAt || fallback || ""));
  return Number.isFinite(value) ? value : 0;
}

function defaultDecisionDetail(): ModelReplayDecisionDetail {
  return {
    bet_label: "$0",
    reason: "No replay decision recorded",
    side: "none",
    team: null,
    stake: 0,
    odds: null,
    model_probability: null,
    market_probability: null,
    edge: null,
    expected_value: null,
    outcome: "no_bet",
    profit: 0,
    payout: 0,
  };
}

function buildEmptyStrategySummaryMap(totalGames: number): Record<ModelReplayStrategyKey, MutableStrategySummary> {
  return {
    riskAdjusted: emptyStrategySummary(totalGames),
    aggressive: emptyStrategySummary(totalGames),
    capitalPreservation: emptyStrategySummary(totalGames),
  };
}

export function buildModelReplayRuns(
  candidates: ModelReplayCandidateRow[],
  runMetadataById: Map<string, ModelReplayRunMetadata>,
  runSummariesById: Map<string, ModelRunSummaryRow>
): ModelReplayRunRow[] {
  const candidatesByRun = new Map<string, ModelReplayCandidateRow[]>();

  for (const row of candidates) {
    if (!row.model_run_id) continue;
    const current = candidatesByRun.get(row.model_run_id) || [];
    current.push(row);
    candidatesByRun.set(row.model_run_id, current);
  }

  return Array.from(candidatesByRun.entries())
    .map(([modelRunId, runRows]) => {
      const metadata = runMetadataById.get(modelRunId);
      const scoredSummary = runSummariesById.get(modelRunId);
      const rowsByDate = new Map<string, ModelReplayCandidateRow[]>();

      for (const row of runRows) {
        const current = rowsByDate.get(row.date_central) || [];
        current.push(row);
        rowsByDate.set(row.date_central, current);
      }

      const strategySummaries = buildEmptyStrategySummaryMap(runRows.length);
      const betRowsByGame = new Map<number, MutableReplayBetRow>();

      for (const [, dayRows] of Array.from(rowsByDate.entries()).sort(([left], [right]) => left.localeCompare(right))) {
        for (const strategy of MODEL_REPLAY_COMPARISON_STRATEGIES) {
          const strategyConfig = getBetStrategyConfig(strategy);
          const decisions = computeBetDecisionsForSlate(
            dayRows.map((row) => ({
              home_team: row.home_team,
              away_team: row.away_team,
              home_win_probability: row.home_win_probability,
              home_moneyline: row.home_moneyline,
              away_moneyline: row.away_moneyline,
              betting_model_name: row.model_name,
              model_win_probabilities: row.model_win_probabilities,
            })),
            strategy,
            strategyConfig,
            strategyConfig.label
          );

          dayRows.forEach((row, index) => {
            const detail = buildDecisionDetail(row, decisions[index]);
            const existing =
              betRowsByGame.get(row.game_id) ||
              {
                game_id: row.game_id,
                date_central: row.date_central,
                forecast_as_of_utc: row.forecast_as_of_utc,
                start_time_utc: row.start_time_utc,
                final_utc: row.final_utc,
                home_team: row.home_team,
                away_team: row.away_team,
                home_score: row.home_score,
                away_score: row.away_score,
                home_moneyline: row.home_moneyline,
                away_moneyline: row.away_moneyline,
                strategies: {},
              };

            existing.strategies[strategy] = detail;
            betRowsByGame.set(row.game_id, existing);
            trackStrategyOutcome(strategySummaries[strategy], detail, row.date_central);
          });
        }
      }

      const bets = Array.from(betRowsByGame.values())
        .sort((left, right) => left.date_central.localeCompare(right.date_central) || left.game_id - right.game_id)
        .map((row) => ({
          ...row,
          strategies: {
            riskAdjusted: row.strategies.riskAdjusted || defaultDecisionDetail(),
            aggressive: row.strategies.aggressive || defaultDecisionDetail(),
            capitalPreservation: row.strategies.capitalPreservation || defaultDecisionDetail(),
          },
        }));

      const featureColumns = metadata?.feature_columns || [];
      const metadataFeatureCount = numberFromMetadata(metadata?.feature_metadata, "n_features");
      const featureCount =
        featureColumns.length || metadataFeatureCount;

      return {
        model_name: metadata?.model_name || scoredSummary?.model_name || runRows[0]?.model_name || "unknown",
        model_run_id: modelRunId,
        run_type: metadata?.run_type || scoredSummary?.run_type || null,
        created_at_utc: metadata?.created_at_utc || scoredSummary?.created_at_utc || null,
        snapshot_id: metadata?.snapshot_id || scoredSummary?.snapshot_id || null,
        artifact_path: metadata?.artifact_path || null,
        feature_set_version: metadata?.feature_set_version || scoredSummary?.feature_set_version || null,
        feature_columns: featureColumns,
        feature_count: featureCount,
        feature_metadata: metadata?.feature_metadata || null,
        params: metadata?.params || null,
        metrics: metadata?.metrics || null,
        scored_games: scoredSummary?.n_games || 0,
        avg_log_loss: typeof scoredSummary?.avg_log_loss === "number" ? scoredSummary.avg_log_loss : null,
        avg_brier: typeof scoredSummary?.avg_brier === "number" ? scoredSummary.avg_brier : null,
        accuracy: typeof scoredSummary?.accuracy === "number" ? scoredSummary.accuracy : null,
        version_rank: typeof scoredSummary?.version_rank === "number" ? scoredSummary.version_rank : null,
        is_latest_version: typeof scoredSummary?.is_latest_version === "number" ? scoredSummary.is_latest_version : null,
        first_game_date_utc: scoredSummary?.first_game_date_utc || null,
        last_game_date_utc: scoredSummary?.last_game_date_utc || null,
        first_replay_date_central: bets[0]?.date_central || null,
        last_replay_date_central: bets[bets.length - 1]?.date_central || null,
        replayable_games: bets.length,
        strategies: {
          riskAdjusted: finalizeStrategySummary(strategySummaries.riskAdjusted),
          aggressive: finalizeStrategySummary(strategySummaries.aggressive),
          capitalPreservation: finalizeStrategySummary(strategySummaries.capitalPreservation),
        },
        bets,
      };
    })
    .sort((left, right) => {
      if (left.model_name !== right.model_name) {
        return left.model_name.localeCompare(right.model_name);
      }
      return sortTimestamp(right.created_at_utc, right.last_replay_date_central) - sortTimestamp(left.created_at_utc, left.last_replay_date_central);
    });
}

function numberFromMetadata(metadata?: TableRow | null, key?: string): number {
  if (!metadata || !key) return 0;
  const value = metadata[key];
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}
