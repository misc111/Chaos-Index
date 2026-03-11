from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  extracted_at_utc TEXT NOT NULL,
  raw_path TEXT NOT NULL,
  metadata_json TEXT,
  freshness_utc TEXT,
  row_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS games (
  game_id INTEGER PRIMARY KEY,
  season INTEGER,
  game_date_utc TEXT,
  start_time_utc TEXT,
  game_state TEXT,
  home_team TEXT,
  away_team TEXT,
  home_team_id INTEGER,
  away_team_id INTEGER,
  venue TEXT,
  is_neutral_site INTEGER DEFAULT 0,
  home_score INTEGER,
  away_score INTEGER,
  went_ot INTEGER DEFAULT 0,
  went_so INTEGER DEFAULT 0,
  home_win INTEGER,
  status_final INTEGER DEFAULT 0,
  as_of_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
  game_id INTEGER PRIMARY KEY,
  season INTEGER,
  game_date_utc TEXT,
  final_utc TEXT,
  home_team TEXT,
  away_team TEXT,
  home_score INTEGER,
  away_score INTEGER,
  home_win INTEGER,
  ingested_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
  team_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  league TEXT NOT NULL,
  team_abbrev TEXT NOT NULL,
  team_name TEXT,
  conference TEXT,
  division TEXT,
  as_of_date TEXT,
  as_of_utc TEXT NOT NULL,
  snapshot_id TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS feature_sets (
  feature_set_version TEXT PRIMARY KEY,
  created_at_utc TEXT NOT NULL,
  snapshot_id TEXT,
  feature_columns_json TEXT NOT NULL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS model_runs (
  model_run_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  run_type TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  snapshot_id TEXT,
  feature_set_version TEXT,
  params_json TEXT,
  metrics_json TEXT,
  artifact_path TEXT,
  model_hash TEXT
);

-- `predictions` is the immutable pregame ledger that powers historical replay.
-- Only live forecasts that truly existed before a game started belong here.
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER NOT NULL,
  as_of_utc TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_run_id TEXT,
  feature_set_version TEXT,
  snapshot_id TEXT,
  game_date_utc TEXT,
  home_team TEXT,
  away_team TEXT,
  prob_home_win REAL NOT NULL,
  pred_winner TEXT,
  prob_low REAL,
  prob_high REAL,
  uncertainty_flags_json TEXT,
  metadata_json TEXT,
  UNIQUE(game_id, as_of_utc, model_name, model_run_id)
);

-- Synthetic diagnostics stay separate so backtests and OOF rows can never
-- silently rewrite the app's view of historical live predictions.
CREATE TABLE IF NOT EXISTS prediction_diagnostics (
  diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER NOT NULL,
  as_of_utc TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_run_id TEXT,
  feature_set_version TEXT,
  snapshot_id TEXT,
  game_date_utc TEXT,
  home_team TEXT,
  away_team TEXT,
  prob_home_win REAL NOT NULL,
  pred_winner TEXT,
  prob_low REAL,
  prob_high REAL,
  uncertainty_flags_json TEXT,
  metadata_json TEXT,
  UNIQUE(game_id, as_of_utc, model_name, model_run_id)
);

CREATE TABLE IF NOT EXISTS upcoming_game_forecasts (
  game_id INTEGER NOT NULL,
  as_of_utc TEXT NOT NULL,
  game_date_utc TEXT,
  home_team TEXT,
  away_team TEXT,
  ensemble_prob_home_win REAL NOT NULL,
  predicted_winner TEXT NOT NULL,
  per_model_probs_json TEXT NOT NULL,
  spread_min REAL,
  spread_median REAL,
  spread_max REAL,
  spread_mean REAL,
  spread_sd REAL,
  spread_iqr REAL,
  bayes_ci_low REAL,
  bayes_ci_high REAL,
  uncertainty_flags_json TEXT,
  snapshot_id TEXT,
  feature_set_version TEXT,
  model_run_id TEXT,
  PRIMARY KEY (game_id, as_of_utc)
);

CREATE TABLE IF NOT EXISTS model_scores (
  score_id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  model_run_id TEXT,
  as_of_utc TEXT NOT NULL,
  game_date_utc TEXT,
  prob_home_win REAL NOT NULL,
  outcome_home_win INTEGER NOT NULL,
  log_loss REAL NOT NULL,
  brier REAL NOT NULL,
  accuracy INTEGER NOT NULL,
  scored_at_utc TEXT NOT NULL,
  UNIQUE(game_id, model_name, as_of_utc, model_run_id)
);

CREATE TABLE IF NOT EXISTS performance_aggregates (
  aggregate_id INTEGER PRIMARY KEY AUTOINCREMENT,
  as_of_utc TEXT NOT NULL,
  model_name TEXT NOT NULL,
  window_label TEXT NOT NULL,
  start_date TEXT,
  end_date TEXT,
  n_games INTEGER NOT NULL,
  log_loss REAL,
  brier REAL,
  accuracy REAL,
  auc REAL,
  ece REAL,
  mce REAL,
  calibration_alpha REAL,
  calibration_beta REAL,
  created_at_utc TEXT NOT NULL,
  UNIQUE(as_of_utc, model_name, window_label)
);

CREATE TABLE IF NOT EXISTS change_points (
  change_id INTEGER PRIMARY KEY AUTOINCREMENT,
  as_of_utc TEXT NOT NULL,
  model_name TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  method TEXT NOT NULL,
  statistic REAL NOT NULL,
  threshold REAL NOT NULL,
  detected INTEGER NOT NULL,
  details_json TEXT
);

CREATE TABLE IF NOT EXISTS validation_results (
  validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  as_of_utc TEXT NOT NULL,
  model_name TEXT,
  validation_name TEXT NOT NULL,
  split_label TEXT,
  result_json TEXT NOT NULL,
  artifact_path TEXT
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
  odds_snapshot_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  league TEXT NOT NULL,
  as_of_utc TEXT NOT NULL,
  raw_path TEXT,
  regions TEXT,
  markets TEXT,
  odds_format TEXT,
  date_format TEXT,
  event_count INTEGER DEFAULT 0,
  row_count INTEGER DEFAULT 0,
  requests_last INTEGER,
  requests_used INTEGER,
  requests_remaining INTEGER,
  from_cache INTEGER DEFAULT 0,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS odds_market_lines (
  line_id INTEGER PRIMARY KEY AUTOINCREMENT,
  odds_snapshot_id TEXT NOT NULL,
  league TEXT NOT NULL,
  game_id INTEGER,
  sport_key TEXT,
  odds_event_id TEXT NOT NULL,
  commence_time_utc TEXT,
  commence_date_central TEXT,
  api_home_team TEXT,
  api_away_team TEXT,
  home_team TEXT,
  away_team TEXT,
  bookmaker_key TEXT,
  bookmaker_title TEXT,
  bookmaker_last_update_utc TEXT,
  market_key TEXT,
  outcome_name TEXT,
  outcome_side TEXT,
  outcome_team TEXT,
  outcome_price REAL,
  outcome_point REAL,
  implied_probability REAL,
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS historical_bet_decisions (
  game_id INTEGER PRIMARY KEY,
  date_central TEXT NOT NULL,
  forecast_as_of_utc TEXT NOT NULL,
  forecast_model_run_id TEXT,
  odds_as_of_utc TEXT NOT NULL,
  odds_snapshot_id TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  home_win_probability REAL NOT NULL,
  home_moneyline REAL NOT NULL,
  away_moneyline REAL NOT NULL,
  bet_label TEXT NOT NULL,
  reason TEXT NOT NULL,
  side TEXT NOT NULL,
  team TEXT,
  stake REAL NOT NULL,
  odds REAL,
  model_probability REAL,
  market_probability REAL,
  edge REAL,
  expected_value REAL,
  decision_logic_version TEXT NOT NULL,
  materialization_version TEXT,
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS historical_bet_decisions_by_profile (
  strategy TEXT NOT NULL,
  sizing_style TEXT NOT NULL,
  game_id INTEGER NOT NULL,
  date_central TEXT NOT NULL,
  forecast_as_of_utc TEXT NOT NULL,
  forecast_model_run_id TEXT,
  odds_as_of_utc TEXT NOT NULL,
  odds_snapshot_id TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  home_win_probability REAL NOT NULL,
  home_moneyline REAL NOT NULL,
  away_moneyline REAL NOT NULL,
  bet_label TEXT NOT NULL,
  reason TEXT NOT NULL,
  side TEXT NOT NULL,
  team TEXT,
  stake REAL NOT NULL,
  odds REAL,
  model_probability REAL,
  market_probability REAL,
  edge REAL,
  expected_value REAL,
  stake_unit_dollars REAL NOT NULL DEFAULT 100,
  strategy_config_signature TEXT,
  decision_logic_version TEXT NOT NULL,
  materialization_version TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (strategy, sizing_style, game_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_game_model ON predictions(game_id, model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_asof ON predictions(as_of_utc);
CREATE INDEX IF NOT EXISTS idx_prediction_diagnostics_game_model ON prediction_diagnostics(game_id, model_name);
CREATE INDEX IF NOT EXISTS idx_prediction_diagnostics_asof ON prediction_diagnostics(as_of_utc);
CREATE INDEX IF NOT EXISTS idx_upcoming_asof ON upcoming_game_forecasts(as_of_utc);
CREATE INDEX IF NOT EXISTS idx_upcoming_asof_date ON upcoming_game_forecasts(as_of_utc, game_date_utc, game_id);
CREATE INDEX IF NOT EXISTS idx_upcoming_asof_home_date ON upcoming_game_forecasts(as_of_utc, home_team, game_date_utc, game_id);
CREATE INDEX IF NOT EXISTS idx_upcoming_asof_away_date ON upcoming_game_forecasts(as_of_utc, away_team, game_date_utc, game_id);
CREATE INDEX IF NOT EXISTS idx_model_scores_model ON model_scores(model_name, scored_at_utc);
CREATE INDEX IF NOT EXISTS idx_model_scores_game_date ON model_scores(game_date_utc);
CREATE INDEX IF NOT EXISTS idx_model_scores_game_model_scoretime ON model_scores(game_id, model_name, scored_at_utc DESC, score_id DESC);
CREATE INDEX IF NOT EXISTS idx_results_final_utc ON results(final_utc);
CREATE INDEX IF NOT EXISTS idx_teams_league_asof ON teams(league, as_of_utc DESC);
CREATE INDEX IF NOT EXISTS idx_teams_league_team_asof ON teams(league, team_abbrev, as_of_utc DESC);
CREATE INDEX IF NOT EXISTS idx_change_points_asof ON change_points(as_of_utc DESC);
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_league_asof ON odds_snapshots(league, as_of_utc DESC);
CREATE INDEX IF NOT EXISTS idx_odds_lines_snapshot ON odds_market_lines(odds_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_odds_lines_league_game_market ON odds_market_lines(league, game_id, market_key, odds_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_odds_lines_event_market_book ON odds_market_lines(odds_event_id, market_key, bookmaker_key);
CREATE INDEX IF NOT EXISTS idx_historical_bet_decisions_date ON historical_bet_decisions(date_central);
CREATE INDEX IF NOT EXISTS idx_historical_bet_decisions_by_profile_date
  ON historical_bet_decisions_by_profile(strategy, sizing_style, date_central);

CREATE TABLE IF NOT EXISTS historical_bet_decisions_by_profile_v2 (
  strategy TEXT NOT NULL,
  sizing_style TEXT NOT NULL,
  strategy_config_signature TEXT NOT NULL DEFAULT '',
  game_id INTEGER NOT NULL,
  date_central TEXT NOT NULL,
  forecast_as_of_utc TEXT NOT NULL,
  forecast_model_run_id TEXT,
  odds_as_of_utc TEXT NOT NULL,
  odds_snapshot_id TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  home_win_probability REAL NOT NULL,
  home_moneyline REAL NOT NULL,
  away_moneyline REAL NOT NULL,
  bet_label TEXT NOT NULL,
  reason TEXT NOT NULL,
  side TEXT NOT NULL,
  team TEXT,
  stake REAL NOT NULL,
  odds REAL,
  model_probability REAL,
  market_probability REAL,
  edge REAL,
  expected_value REAL,
  stake_unit_dollars REAL NOT NULL DEFAULT 100,
  decision_logic_version TEXT NOT NULL,
  materialization_version TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (strategy, sizing_style, strategy_config_signature, game_id)
);

CREATE INDEX IF NOT EXISTS idx_historical_bet_decisions_by_profile_v2_date
  ON historical_bet_decisions_by_profile_v2(strategy, sizing_style, strategy_config_signature, date_central);
"""
