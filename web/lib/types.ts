export type TableRow = Record<string, unknown>;

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
  moneyline: MarketMoneylineQuote;
  spread: MarketSpreadQuote;
  total: MarketTotalQuote;
};

export type MarketBoardResponse = {
  league: string;
  as_of_utc?: string | null;
  odds_as_of_utc?: string | null;
  date_central?: string;
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
  change_points: ChangePointRow[];
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
  replay_decisions?: Record<string, unknown> | null;
};

export type GamesTodayResponse = {
  league?: string;
  as_of_utc?: string | null;
  odds_as_of_utc?: string | null;
  date_central?: string;
  historical_coverage_start_central?: string | null;
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
