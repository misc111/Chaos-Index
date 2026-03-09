import type { BetStrategy } from "@/lib/betting-strategy";
import type { BetStrategyOptimizationSummary, ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import type { LeagueCode } from "@/lib/league";

export type HistoricalBetRow = {
  game_id: number;
  date_central: string;
  week_start_central: string;
  start_time_utc: string | null;
  final_utc: string | null;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  forecast_as_of_utc: string;
  odds_as_of_utc: string;
  odds_snapshot_id: string;
  home_moneyline: number;
  away_moneyline: number;
  bet_label: string;
  reason: string;
  side: "home" | "away";
  team: string;
  stake: number;
  odds: number;
  expected_value: number | null;
  edge: number | null;
  model_probability: number | null;
  market_probability: number | null;
  outcome: "win" | "loss";
  profit: number;
  payout: number;
  cumulative_profit: number;
};

export type HistoricalDailyPoint = {
  date_central: string;
  risked: number;
  daily_profit: number;
  cumulative_profit: number;
  cumulative_bankroll: number;
  bet_count: number;
};

export type BetHistorySummary = {
  total_final_games: number;
  games_with_forecast: number;
  games_with_odds: number;
  analyzed_games: number;
  suggested_bets: number;
  wins: number;
  losses: number;
  total_risked: number;
  total_profit: number;
  roi: number;
  starting_bankroll: number;
  current_bankroll: number;
  bankroll_start_central: string | null;
  coverage_start_central: string | null;
  coverage_end_central: string | null;
  note: string;
};

export type BetHistoryStrategyBundle = {
  summary: BetHistorySummary;
  daily_points: HistoricalDailyPoint[];
  bets: HistoricalBetRow[];
};

export type BetHistoryResponse = {
  league: LeagueCode;
  default_strategy: BetStrategy;
  strategy_configs: Record<BetStrategy, ResolvedBetStrategyConfig>;
  strategy_optimization: BetStrategyOptimizationSummary;
  strategies: Record<BetStrategy, BetHistoryStrategyBundle>;
};
