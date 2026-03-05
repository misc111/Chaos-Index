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

export type PredictionsResponse = {
  league: string;
  as_of_utc?: string;
  model_columns: string[];
  model_trust_notes: Record<string, string>;
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
