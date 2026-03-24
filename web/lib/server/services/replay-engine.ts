import { settleBet, type BetDecision } from "@/lib/betting";
import type { ModelReplayDecisionDetail, ModelReplayStrategySummary } from "@/lib/types";

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
