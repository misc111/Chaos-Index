import { computeBetDecisionsForSlate, settleBet, type BetDecision } from "@/lib/betting";
import { getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";
import type { LeagueCode } from "@/lib/league";
import type { ModelReplayBetRow, ModelReplayDecisionDetail, ModelReplayStrategySummary } from "@/lib/types";

export type MutableReplayStrategySummary = {
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

export type ReplayDecisionRowCore = {
  game_id: number;
  date_central: string;
  forecast_as_of_utc: string;
  start_time_utc: string | null;
  final_utc: string | null;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_moneyline: number;
  away_moneyline: number;
  home_win: number | null;
  home_win_probability: number;
  model_win_probabilities: Record<string, number | null>;
};

export type MutableReplayBetRow = Omit<ModelReplayBetRow, "strategies"> & {
  strategies: Partial<Record<BetStrategy, ModelReplayDecisionDetail>>;
};

export function createEmptyReplayStrategySummary(totalGames: number): MutableReplayStrategySummary {
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

export function finalizeReplayStrategySummary(summary: MutableReplayStrategySummary): ModelReplayStrategySummary {
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

export function buildReplayDecisionDetail(decision: BetDecision, homeWin: number | null): ModelReplayDecisionDetail {
  const settlement = settleBet(decision, homeWin);
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

export function createEmptyReplayStrategySummaryMap(totalGames: number): Record<BetStrategy, MutableReplayStrategySummary> {
  return {
    riskAdjusted: createEmptyReplayStrategySummary(totalGames),
    aggressive: createEmptyReplayStrategySummary(totalGames),
    capitalPreservation: createEmptyReplayStrategySummary(totalGames),
  };
}

export function groupReplayRowsByDate<Row extends { date_central: string }>(rows: Row[]): Array<[string, Row[]]> {
  const rowsByDate = new Map<string, Row[]>();
  for (const row of rows) {
    const current = rowsByDate.get(row.date_central) || [];
    current.push(row);
    rowsByDate.set(row.date_central, current);
  }
  return Array.from(rowsByDate.entries()).sort(([left], [right]) => left.localeCompare(right));
}

export function ensureReplayBetRow<Row extends ReplayDecisionRowCore>(
  betRowsByGame: Map<number, MutableReplayBetRow>,
  row: Row
): MutableReplayBetRow {
  const existing = betRowsByGame.get(row.game_id);
  if (existing) {
    return existing;
  }

  const created: MutableReplayBetRow = {
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
  betRowsByGame.set(row.game_id, created);
  return created;
}

export function evaluateReplayStrategyDecisionsForDay<Row extends ReplayDecisionRowCore>(
  dayRows: Row[],
  options: {
    league: LeagueCode;
    strategies: readonly BetStrategy[];
    bettingModelNameForRow: (row: Row) => string;
    onDecision: (row: Row, strategy: BetStrategy, detail: ModelReplayDecisionDetail) => void;
  }
): void {
  for (const strategy of options.strategies) {
    const strategyConfig = getBetStrategyConfig(strategy, { league: options.league });
    const decisions = computeBetDecisionsForSlate(
      dayRows.map((row) => ({
        league: options.league,
        home_team: row.home_team,
        away_team: row.away_team,
        home_win_probability: row.home_win_probability,
        home_moneyline: row.home_moneyline,
        away_moneyline: row.away_moneyline,
        betting_model_name: options.bettingModelNameForRow(row),
        model_win_probabilities: row.model_win_probabilities,
      })),
      strategy,
      strategyConfig,
      strategyConfig.label
    );

    dayRows.forEach((row, index) => {
      const detail = buildReplayDecisionDetail(decisions[index], row.home_win);
      options.onDecision(row, strategy, detail);
    });
  }
}

export function trackReplayStrategyOutcome(
  summary: MutableReplayStrategySummary,
  detail: ModelReplayDecisionDetail,
  dateCentral: string
): void {
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

export function finalizeReplayBetRows(
  betRowsByGame: Map<number, MutableReplayBetRow>,
  sortFn: (left: MutableReplayBetRow, right: MutableReplayBetRow) => number
): Array<
  Omit<MutableReplayBetRow, "strategies"> & {
    strategies: Record<BetStrategy, ModelReplayDecisionDetail>;
  }
> {
  return Array.from(betRowsByGame.values())
    .sort(sortFn)
    .map((row) => ({
      ...row,
      strategies: {
        riskAdjusted: row.strategies.riskAdjusted || defaultReplayDecisionDetail(),
        aggressive: row.strategies.aggressive || defaultReplayDecisionDetail(),
        capitalPreservation: row.strategies.capitalPreservation || defaultReplayDecisionDetail(),
      },
    }));
}

export function defaultReplayDecisionDetail(): ModelReplayDecisionDetail {
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
