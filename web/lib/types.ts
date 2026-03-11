import type { HistoricalReplayDecisionSet } from "@/lib/replay-bets";
import type { BetStrategyOptimizationSummary, ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import type { BetStrategy } from "@/lib/betting-strategy";
import type { ModelWinProbabilities } from "@/lib/betting-model";

export type TableRow = Record<string, unknown>;
export type ReplayStrategyMap<T> = Record<BetStrategy, T>;

export type ForecastRow = {
  game_id: number;
  game_date_utc: string;
  home_team: string;
  away_team: string;
  ensemble_prob_home_win: number;
  predicted_winner: string;
  spread_mean?: number;
  spread_sd?: number;
  bayes_ci_low?: number;
  bayes_ci_high?: number;
  uncertainty_flags_json?: string;
  model_win_probabilities?: Record<string, number | null>;
  odds_as_of_utc?: string;
  home_moneyline?: number;
  away_moneyline?: number;
  moneyline_book?: string;
};

export type PredictionModelSummary = {
  headline?: string;
  trust_note: string;
  active_feature_count?: number;
  active_features?: string[];
};

export type PredictionsResponse = {
  league: string;
  as_of_utc?: string;
  model_columns: string[];
  model_trust_notes: Record<string, string>;
  model_summaries: Record<string, PredictionModelSummary>;
  model_feature_map_updated_at_utc?: string;
  rows: ForecastRow[];
};

export type MarketMoneylineQuote = {
  away_price?: number | null;
  home_price?: number | null;
  away_book?: string | null;
  home_book?: string | null;
  books_count?: number;
};

export type MarketSpreadQuote = {
  point?: number | null;
  away_price?: number | null;
  home_price?: number | null;
  away_book?: string | null;
  home_book?: string | null;
  books_count?: number;
};

export type MarketTotalQuote = {
  point?: number | null;
  over_price?: number | null;
  under_price?: number | null;
  over_book?: string | null;
  under_book?: string | null;
  books_count?: number;
};

export type MarketBoardRow = {
  game_id: number;
  game_date_utc?: string | null;
  start_time_utc?: string | null;
  home_team: string;
  away_team: string;
  home_team_name: string;
  away_team_name: string;
  home_win_probability: number;
  betting_model_name?: string | null;
  model_win_probabilities?: ModelWinProbabilities | null;
  moneyline: MarketMoneylineQuote;
  spread: MarketSpreadQuote;
  total: MarketTotalQuote;
};

export type MarketBoardResponse = {
  league: string;
  as_of_utc?: string | null;
  odds_as_of_utc?: string | null;
  date_central?: string;
  strategy_configs?: Record<BetStrategy, ResolvedBetStrategyConfig>;
  rows: MarketBoardRow[];
};

export type LeaderboardRow = {
  model_name: string;
  window_label: string;
  n_games: number;
  log_loss: number;
  brier: number;
  accuracy: number;
  ece: number;
  calibration_alpha: number;
  calibration_beta: number;
};

export type PerformanceScoreRow = {
  model_name: string;
  game_date_utc: string;
  log_loss: number;
};

export type ModelRunSummaryRow = {
  model_name: string;
  model_run_id: string;
  run_type?: string | null;
  created_at_utc?: string | null;
  snapshot_id?: string | null;
  feature_set_version?: string | null;
  first_game_date_utc?: string | null;
  last_game_date_utc?: string | null;
  n_games: number;
  avg_log_loss: number;
  avg_brier: number;
  accuracy: number;
  version_rank: number;
  is_latest_version: number;
};

export type ModelReplayStrategySummary = {
  total_games: number;
  suggested_bets: number;
  wins: number;
  losses: number;
  total_risked: number;
  total_profit: number;
  roi: number;
  avg_edge: number | null;
  avg_expected_value: number | null;
  first_bet_date_central: string | null;
  last_bet_date_central: string | null;
};

export type ModelReplayDecisionDetail = {
  bet_label: string;
  reason: string;
  side: "home" | "away" | "none";
  team: string | null;
  stake: number;
  odds: number | null;
  model_probability: number | null;
  market_probability: number | null;
  edge: number | null;
  expected_value: number | null;
  outcome: "win" | "loss" | "no_bet";
  profit: number;
  payout: number;
};

export type ModelReplayBetRow = {
  game_id: number;
  date_central: string;
  forecast_as_of_utc: string;
  start_time_utc?: string | null;
  final_utc?: string | null;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_moneyline: number;
  away_moneyline: number;
  strategies: ReplayStrategyMap<ModelReplayDecisionDetail>;
};

export type ModelReplayRunRow = {
  model_name: string;
  model_run_id: string;
  run_type?: string | null;
  created_at_utc?: string | null;
  snapshot_id?: string | null;
  artifact_path?: string | null;
  feature_set_version?: string | null;
  feature_columns: string[];
  feature_count: number;
  feature_metadata?: TableRow | null;
  params?: TableRow | null;
  metrics?: TableRow | null;
  scored_games: number;
  avg_log_loss: number | null;
  avg_brier: number | null;
  accuracy: number | null;
  version_rank: number | null;
  is_latest_version: number | null;
  first_game_date_utc?: string | null;
  last_game_date_utc?: string | null;
  first_replay_date_central: string | null;
  last_replay_date_central: string | null;
  replayable_games: number;
  strategies: ReplayStrategyMap<ModelReplayStrategySummary>;
  bets: ModelReplayBetRow[];
};

export type SnapshotCommitInfo = {
  sha: string;
  short_sha: string;
  committed_at_utc: string;
  subject: string;
};

export type EnsembleSnapshotComponentModelRow = {
  model_name: string;
  selected_for_training: number;
  included_in_ensemble: number;
  demoted_from_ensemble: number;
  weight: number | null;
  feature_columns: string[];
  feature_count: number;
  train_metrics?: TableRow | null;
  train_params?: TableRow | null;
};

export type EnsembleSnapshotDailyStrategyRow = {
  slate_games: number;
  suggested_bets: number;
  wins: number;
  losses: number;
  total_risked: number;
  total_profit: number;
  cumulative_risked: number;
  cumulative_profit: number;
  roi: number;
  cumulative_roi: number;
};

export type EnsembleSnapshotDailyRow = {
  date_central: string;
  slate_games: number;
  strategies: ReplayStrategyMap<EnsembleSnapshotDailyStrategyRow>;
};

export type EnsembleSnapshotRow = {
  snapshot_key: string;
  model_name: string;
  model_run_id: string;
  ensemble_model_run_id: string;
  finalized_at_utc?: string | null;
  finalized_date_central: string | null;
  activation_date_central: string;
  compared_through_date_central: string | null;
  pregame_cutoff_utc?: string | null;
  snapshot_id?: string | null;
  artifact_path?: string | null;
  feature_set_version?: string | null;
  calibration_fingerprint: string;
  feature_columns: string[];
  feature_count: number;
  feature_metadata?: TableRow | null;
  params?: TableRow | null;
  metrics?: TableRow | null;
  tuning?: TableRow | null;
  selected_models: string[];
  ensemble_component_columns: string[];
  demoted_models: string[];
  stack_base_columns: string[];
  glm_feature_columns: string[];
  model_feature_columns?: Record<string, string[]> | null;
  component_models: EnsembleSnapshotComponentModelRow[];
  model_commit?: SnapshotCommitInfo | null;
  commit_window: SnapshotCommitInfo[];
  replayable_games: number;
  days_tracked: number;
  strategies: ReplayStrategyMap<ModelReplayStrategySummary>;
  daily: EnsembleSnapshotDailyRow[];
  bets: ModelReplayBetRow[];
};

export type ChangePointRow = TableRow & {
  model_name?: string;
  metric_name?: string;
  method?: string;
  statistic?: number;
  threshold?: number;
  details_json?: string;
  as_of_utc?: string;
};

export type PerformanceResponse = {
  league: string;
  scores: PerformanceScoreRow[];
  run_summaries: ModelRunSummaryRow[];
  change_points: ChangePointRow[];
  replay_runs: ModelReplayRunRow[];
  ensemble_snapshots: EnsembleSnapshotRow[];
  default_replay_strategy?: BetStrategy;
  comparison_replay_strategy?: BetStrategy;
};

export type ValidationSections = Record<string, TableRow[]>;

export type ValidationResponse = {
  league?: string;
  significance?: TableRow[];
  sections: ValidationSections;
};

export type MetricsResponse = {
  league?: string;
  leaderboard: LeaderboardRow[];
  calibration: TableRow[];
  slices: TableRow[];
};

export type GamesTodayRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  betting_model_name?: string | null;
  model_win_probabilities?: ModelWinProbabilities | null;
  forecast_as_of_utc?: string | null;
  odds_as_of_utc?: string | null;
  start_time_utc?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  home_moneyline_book?: string | null;
  away_moneyline_book?: string | null;
  over_190_price?: number | null;
  over_190_point?: number | null;
  over_190_book?: string | null;
  replay_decisions?: HistoricalReplayDecisionSet | null;
};

export type GamesTodayResponse = {
  league?: string;
  as_of_utc?: string | null;
  odds_as_of_utc?: string | null;
  date_central?: string;
  historical_coverage_start_central?: string | null;
  strategy_configs?: Record<BetStrategy, ResolvedBetStrategyConfig>;
  strategy_optimization?: BetStrategyOptimizationSummary;
  historical_rows?: GamesTodayRow[];
  rows?: GamesTodayRow[];
};

export type RefreshOddsResponse = {
  ok?: boolean;
  error?: string;
  details?: string;
  odds_as_of_utc?: string | null;
  event_count?: number | null;
  row_count?: number | null;
};

export type ActualVsExpectedHistoricalRow = {
  game_id: number;
  game_date_utc: string;
  home_team: string;
  away_team: string;
  as_of_utc: string;
  prob_home_win: number;
  predicted_winner: string;
  home_win: number;
  final_utc?: string | null;
  start_time_utc?: string | null;
  is_toss_up?: number;
  model_correct?: number | null;
};

export type ActualVsExpectedUpcomingRow = {
  game_id: number;
  game_date_utc: string;
  home_team: string;
  away_team: string;
  as_of_utc: string;
  ensemble_prob_home_win: number;
  predicted_winner: string;
  start_time_utc?: string | null;
};

export type ActualVsExpectedResponse = {
  league?: string;
  as_of_utc?: string;
  historical_rows?: ActualVsExpectedHistoricalRow[];
  upcoming_rows?: ActualVsExpectedUpcomingRow[];
};
