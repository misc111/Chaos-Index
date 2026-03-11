import { computeBetDecisionsForSlate, settleBet } from "@/lib/betting";
import { BET_STRATEGIES, getBetStrategyConfig } from "@/lib/betting-strategy";
import type {
  EnsembleSnapshotComponentModelRow,
  EnsembleSnapshotDailyRow,
  EnsembleSnapshotRow,
  ModelReplayBetRow,
  ModelReplayDecisionDetail,
  ModelReplayStrategySummary,
  SnapshotCommitInfo,
  TableRow,
} from "@/lib/types";

type EnsembleSnapshotStrategyKey = keyof EnsembleSnapshotRow["strategies"];

export const ENSEMBLE_SNAPSHOT_COMPARISON_STRATEGIES: readonly EnsembleSnapshotStrategyKey[] = BET_STRATEGIES;

export type EnsembleSnapshotCandidateRow = {
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
  model_run_id: string;
  home_win_probability: number;
  model_win_probabilities: Record<string, number | null>;
};

export type EnsembleSnapshotSelection = {
  activation_date_central: string;
  pregame_cutoff_utc: string;
  model_run_id: string;
};

export type EnsembleSnapshotRunMetadata = {
  model_name: string;
  model_run_id: string;
  ensemble_model_run_id: string;
  finalized_at_utc?: string | null;
  finalized_date_central: string | null;
  snapshot_id?: string | null;
  artifact_path?: string | null;
  feature_set_version?: string | null;
  calibration_fingerprint: string;
  feature_columns: string[];
  feature_metadata?: TableRow | null;
  params?: TableRow | null;
  metrics?: TableRow | null;
  tuning?: TableRow | null;
  selected_models: string[];
  ensemble_component_columns: string[];
  demoted_models: string[];
  stack_base_columns: string[];
  glm_feature_columns: string[];
  model_feature_columns: Record<string, string[]>;
  component_models: EnsembleSnapshotComponentModelRow[];
  model_commit?: SnapshotCommitInfo | null;
  commit_window: SnapshotCommitInfo[];
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
  row: EnsembleSnapshotCandidateRow,
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

function buildEmptyStrategySummaryMap(totalGames: number): Record<EnsembleSnapshotStrategyKey, MutableStrategySummary> {
  return {
    riskAdjusted: emptyStrategySummary(totalGames),
    aggressive: emptyStrategySummary(totalGames),
    capitalPreservation: emptyStrategySummary(totalGames),
  };
}

function buildEmptyCumulativeState(): Record<EnsembleSnapshotStrategyKey, { total_profit: number; total_risked: number }> {
  return {
    riskAdjusted: { total_profit: 0, total_risked: 0 },
    aggressive: { total_profit: 0, total_risked: 0 },
    capitalPreservation: { total_profit: 0, total_risked: 0 },
  };
}

function sortTimestamp(value?: string | null, fallbackDate?: string | null): number {
  const fallback = fallbackDate ? `${fallbackDate}T23:59:59Z` : "";
  const numeric = Date.parse(String(value || fallback || ""));
  return Number.isFinite(numeric) ? numeric : 0;
}

function buildDailyStrategyRow(
  totalGames: number,
  daySummary: MutableStrategySummary,
  cumulativeProfit: number,
  cumulativeRisked: number
): EnsembleSnapshotDailyRow["strategies"]["riskAdjusted"] {
  return {
    slate_games: totalGames,
    suggested_bets: daySummary.suggested_bets,
    wins: daySummary.wins,
    losses: daySummary.losses,
    total_risked: daySummary.total_risked,
    total_profit: daySummary.total_profit,
    cumulative_risked: cumulativeRisked,
    cumulative_profit: cumulativeProfit,
    roi: daySummary.total_risked > 0 ? daySummary.total_profit / daySummary.total_risked : 0,
    cumulative_roi: cumulativeRisked > 0 ? cumulativeProfit / cumulativeRisked : 0,
  };
}

export function buildEnsembleSnapshots(
  candidates: EnsembleSnapshotCandidateRow[],
  selections: EnsembleSnapshotSelection[],
  runMetadataById: Map<string, EnsembleSnapshotRunMetadata>
): EnsembleSnapshotRow[] {
  const rowsByRun = new Map<string, EnsembleSnapshotCandidateRow[]>();

  for (const row of candidates) {
    const current = rowsByRun.get(row.model_run_id) || [];
    current.push(row);
    rowsByRun.set(row.model_run_id, current);
  }

  return selections
    .map((selection) => {
      const metadata = runMetadataById.get(selection.model_run_id);
      const runRows = (rowsByRun.get(selection.model_run_id) || [])
        .filter((row) => row.date_central >= selection.activation_date_central)
        .slice()
        .sort(
          (left, right) =>
            left.date_central.localeCompare(right.date_central) ||
            sortTimestamp(left.start_time_utc, left.date_central) - sortTimestamp(right.start_time_utc, right.date_central) ||
            left.game_id - right.game_id
        );

      const strategyTotals = buildEmptyStrategySummaryMap(runRows.length);
      const cumulativeState = buildEmptyCumulativeState();
      const dailyRows: EnsembleSnapshotDailyRow[] = [];
      const betRowsByGame = new Map<number, MutableReplayBetRow>();
      const dayRowsByDate = new Map<string, EnsembleSnapshotCandidateRow[]>();

      for (const row of runRows) {
        const current = dayRowsByDate.get(row.date_central) || [];
        current.push(row);
        dayRowsByDate.set(row.date_central, current);
      }

      // Each snapshot is intentionally replayed from the date it first became
      // the last truthful pregame model state. We never "upgrade" the model
      // midstream inside this loop; the whole point is to ask how this one
      // frozen snapshot would have behaved if we had stopped recalibrating.
      for (const [dateCentral, dayRows] of Array.from(dayRowsByDate.entries()).sort(([left], [right]) => left.localeCompare(right))) {
        const daySummaries = buildEmptyStrategySummaryMap(dayRows.length);

        for (const strategy of ENSEMBLE_SNAPSHOT_COMPARISON_STRATEGIES) {
          const strategyConfig = getBetStrategyConfig(strategy);
          const decisions = computeBetDecisionsForSlate(
            dayRows.map((row) => ({
              home_team: row.home_team,
              away_team: row.away_team,
              home_win_probability: row.home_win_probability,
              home_moneyline: row.home_moneyline,
              away_moneyline: row.away_moneyline,
              betting_model_name: "ensemble",
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
            trackStrategyOutcome(daySummaries[strategy], detail, dateCentral);
            trackStrategyOutcome(strategyTotals[strategy], detail, dateCentral);
          });
        }

        for (const strategy of ENSEMBLE_SNAPSHOT_COMPARISON_STRATEGIES) {
          cumulativeState[strategy].total_profit += daySummaries[strategy].total_profit;
          cumulativeState[strategy].total_risked += daySummaries[strategy].total_risked;
        }

        dailyRows.push({
          date_central: dateCentral,
          slate_games: dayRows.length,
          strategies: {
            riskAdjusted: buildDailyStrategyRow(
              dayRows.length,
              daySummaries.riskAdjusted,
              cumulativeState.riskAdjusted.total_profit,
              cumulativeState.riskAdjusted.total_risked
            ),
            aggressive: buildDailyStrategyRow(
              dayRows.length,
              daySummaries.aggressive,
              cumulativeState.aggressive.total_profit,
              cumulativeState.aggressive.total_risked
            ),
            capitalPreservation: buildDailyStrategyRow(
              dayRows.length,
              daySummaries.capitalPreservation,
              cumulativeState.capitalPreservation.total_profit,
              cumulativeState.capitalPreservation.total_risked
            ),
          },
        });
      }

      const bets = Array.from(betRowsByGame.values())
        .sort(
          (left, right) =>
            left.date_central.localeCompare(right.date_central) ||
            sortTimestamp(left.start_time_utc, left.date_central) - sortTimestamp(right.start_time_utc, right.date_central) ||
            left.game_id - right.game_id
        )
        .map((row) => ({
          ...row,
          strategies: {
            riskAdjusted: row.strategies.riskAdjusted || defaultDecisionDetail(),
            aggressive: row.strategies.aggressive || defaultDecisionDetail(),
            capitalPreservation: row.strategies.capitalPreservation || defaultDecisionDetail(),
          },
        }));

      const featureColumns = metadata?.feature_columns || [];
      const featureCount = featureColumns.length || Number(metadata?.feature_metadata?.n_features || 0) || 0;

      return {
        snapshot_key: `${selection.activation_date_central}::${selection.model_run_id}`,
        model_name: metadata?.model_name || "ensemble",
        model_run_id: selection.model_run_id,
        ensemble_model_run_id: metadata?.ensemble_model_run_id || `${selection.model_run_id}__ensemble`,
        finalized_at_utc: metadata?.finalized_at_utc || null,
        finalized_date_central: metadata?.finalized_date_central || null,
        activation_date_central: selection.activation_date_central,
        compared_through_date_central: dailyRows[dailyRows.length - 1]?.date_central || null,
        pregame_cutoff_utc: selection.pregame_cutoff_utc,
        snapshot_id: metadata?.snapshot_id || null,
        artifact_path: metadata?.artifact_path || null,
        feature_set_version: metadata?.feature_set_version || null,
        calibration_fingerprint: metadata?.calibration_fingerprint || "untracked",
        feature_columns: featureColumns,
        feature_count: featureCount,
        feature_metadata: metadata?.feature_metadata || null,
        params: metadata?.params || null,
        metrics: metadata?.metrics || null,
        tuning: metadata?.tuning || null,
        selected_models: metadata?.selected_models || [],
        ensemble_component_columns: metadata?.ensemble_component_columns || [],
        demoted_models: metadata?.demoted_models || [],
        stack_base_columns: metadata?.stack_base_columns || [],
        glm_feature_columns: metadata?.glm_feature_columns || [],
        model_feature_columns: metadata?.model_feature_columns || {},
        component_models: metadata?.component_models || [],
        model_commit: metadata?.model_commit || null,
        commit_window: metadata?.commit_window || [],
        replayable_games: bets.length,
        days_tracked: dailyRows.length,
        strategies: {
          riskAdjusted: finalizeStrategySummary(strategyTotals.riskAdjusted),
          aggressive: finalizeStrategySummary(strategyTotals.aggressive),
          capitalPreservation: finalizeStrategySummary(strategyTotals.capitalPreservation),
        },
        daily: dailyRows,
        bets,
      };
    })
    .sort(
      (left, right) =>
        sortTimestamp(right.finalized_at_utc, right.activation_date_central) -
        sortTimestamp(left.finalized_at_utc, left.activation_date_central)
    );
}
