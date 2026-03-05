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
