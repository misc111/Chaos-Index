import numpy as np
import pandas as pd

from src.training.model_feature_research import (
    _model_feature_pruning_config,
    load_model_feature_map,
    research_model_feature_map,
)


def test_glm_pruning_config_is_league_specific() -> None:
    assert _model_feature_pruning_config("glm_ridge", league="MLB") == (8, 14, 0.92)
    assert _model_feature_pruning_config("glm_ridge", league="NBA") == (6, 10, 0.92)
    assert _model_feature_pruning_config("glm_ridge", league="NHL") == (14, 24, 0.92)
    assert _model_feature_pruning_config("glm_elastic_net", league="MLB") == (8, 14, 0.92)
    assert _model_feature_pruning_config("glm_elastic_net", league="NBA") == (6, 10, 0.92)
    assert _model_feature_pruning_config("glm_elastic_net", league="NHL") == (14, 24, 0.92)
    assert _model_feature_pruning_config("glm_lasso", league="MLB") == (6, 10, 0.92)
    assert _model_feature_pruning_config("glm_lasso", league="NBA") == (6, 10, 0.92)
    assert _model_feature_pruning_config("glm_lasso", league="NHL") == (14, 24, 0.92)


def test_mlb_model_feature_research_prefers_baseball_features_and_excludes_hockey_prefixes(tmp_path) -> None:
    n = 260
    rng = np.random.default_rng(84)
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": pd.date_range("2026-03-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2026-03-01", periods=n, freq="D").date.astype(str),
            "home_win": (skill + 0.2 * noise > 0).astype(int),
            "diff_starting_pitcher_quality": 1.1 * skill + rng.normal(0, 0.1, n),
            "starting_pitcher_hand_matchup": 0.7 * skill + rng.normal(0, 0.15, n),
            "diff_bullpen_quality": 0.95 * skill + rng.normal(0, 0.15, n),
            "diff_bullpen_fatigue": -0.5 * skill + rng.normal(0, 0.2, n),
            "diff_lineup_talent": 0.9 * skill + rng.normal(0, 0.15, n),
            "diff_lineup_availability": 0.55 * skill + rng.normal(0, 0.2, n),
            "lineup_stability_diff": 0.45 * skill + rng.normal(0, 0.2, n),
            "diff_offense_form": 0.8 * skill + rng.normal(0, 0.15, n),
            "diff_pitching_form": 0.78 * skill + rng.normal(0, 0.15, n),
            "park_run_effect": 0.5 * skill + rng.normal(0, 0.15, n),
            "park_weather_run_effect": 0.42 * skill + rng.normal(0, 0.15, n),
            "weather_wind_run_effect": 0.35 * skill + rng.normal(0, 0.15, n),
            "weather_temperature_run_effect": 0.22 * skill + rng.normal(0, 0.18, n),
            "weather_humidity_run_effect": 0.18 * skill + rng.normal(0, 0.18, n),
            "umpire_run_effect": 0.2 * skill + rng.normal(0, 0.18, n),
            "rest_diff": 0.3 * skill + rng.normal(0, 0.2, n),
            "travel_diff": -0.25 * skill + rng.normal(0, 0.2, n),
            "home_field_advantage": 0.15 + rng.normal(0, 0.03, n),
            "series_context_index": 0.28 * skill + rng.normal(0, 0.2, n),
            "day_game_indicator": rng.integers(0, 2, n),
            "home_days_into_season": np.linspace(1, 162, n),
            "away_days_into_season": np.linspace(1, 162, n) + rng.normal(0, 0.1, n),
            "home_bullpen_quality_index": 0.7 * skill + rng.normal(0, 0.2, n),
            "away_bullpen_quality_index": -0.6 * skill + rng.normal(0, 0.2, n),
            "home_starting_pitcher_quality_index": 0.9 * skill + rng.normal(0, 0.2, n),
            "away_starting_pitcher_quality_index": -0.8 * skill + rng.normal(0, 0.2, n),
            "home_starting_pitcher_hand": rng.integers(0, 2, n),
            "away_starting_pitcher_hand": rng.integers(0, 2, n),
            "goalie_quality_diff": 0.85 * skill + rng.normal(0, 0.15, n),
            "special_pp_diff": 0.8 * skill + rng.normal(0, 0.15, n),
            "rink_goal_effect": 0.75 * skill + rng.normal(0, 0.15, n),
            "target_total_runs": 8.5 + 1.4 * skill + rng.normal(0, 0.4, n),
            "target_run_differential": 0.8 * skill + rng.normal(0, 0.25, n),
        }
    )

    result = research_model_feature_map(
        df,
        league="MLB",
        artifacts_dir=str(tmp_path / "artifacts"),
        feature_columns=[c for c in df.columns if c not in {"game_id", "start_time_utc", "game_date_utc", "home_win"}],
        selected_models=["glm_ridge", "two_stage"],
        approve_changes=True,
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
    )

    saved = load_model_feature_map("MLB", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    assert result.registry_updated is True
    assert "glm_ridge" in saved
    assert "two_stage" in saved
    assert "diff_starting_pitcher_quality" in saved["glm_ridge"]
    assert "diff_bullpen_quality" in saved["glm_ridge"]
    assert "park_run_effect" in saved["glm_ridge"]
    assert "starting_pitcher_hand_matchup" in saved["two_stage"]
    assert "series_context_index" in saved["two_stage"]
    assert len(saved["glm_ridge"]) <= 14
    assert len(saved["two_stage"]) <= 20
    assert not any(feature.startswith(("goalie_", "special_", "rink_")) for feature in saved["glm_ridge"])
    assert not any(feature.startswith(("goalie_", "special_", "rink_")) for feature in saved["two_stage"])


def test_nba_model_feature_research_promotes_per_model_feature_map(tmp_path) -> None:
    n = 260
    rng = np.random.default_rng(42)
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": pd.date_range("2025-10-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-10-01", periods=n, freq="D").date.astype(str),
            "home_win": (skill + 0.3 * noise > 0).astype(int),
            "diff_form_point_margin": 1.1 * skill + rng.normal(0, 0.1, n),
            "diff_form_win_rate": 0.9 * skill + rng.normal(0, 0.1, n),
            "travel_diff": 0.2 * skill + rng.normal(0, 0.2, n),
            "rest_diff": 0.25 * skill + rng.normal(0, 0.2, n),
            "discipline_free_throw_pressure_diff": 0.8 * skill + rng.normal(0, 0.15, n),
            "discipline_foul_margin_diff": 0.7 * skill + rng.normal(0, 0.15, n),
            "availability_stress_diff": 0.5 * skill + rng.normal(0, 0.2, n),
            "arena_margin_effect": 0.4 * skill + rng.normal(0, 0.2, n),
            "elo_home_prob": 0.55 + 0.1 * np.tanh(skill),
            "dyn_home_prob": 0.55 + 0.12 * np.tanh(skill),
            "home_ewm_points_for": 110 + 5 * skill + rng.normal(0, 1, n),
            "away_ewm_points_for": 108 - 4 * skill + rng.normal(0, 1, n),
            "home_ewm_point_margin": 6 * skill + rng.normal(0, 0.5, n),
            "away_ewm_point_margin": -5 * skill + rng.normal(0, 0.5, n),
            "home_ewm_shot_volume_share": 0.52 + 0.04 * np.tanh(skill),
            "away_ewm_shot_volume_share": 0.48 - 0.04 * np.tanh(skill),
            "home_ewm_free_throw_pressure": 0.20 + 0.02 * np.tanh(skill),
            "away_ewm_free_throw_pressure": 0.18 - 0.02 * np.tanh(skill),
            "home_ewm_possession_proxy": 180 + 4 * skill + rng.normal(0, 1, n),
            "away_ewm_possession_proxy": 178 - 3 * skill + rng.normal(0, 1, n),
            "target_shot_volume_share": 0.5 + 0.08 * np.tanh(skill),
            "target_free_throw_pressure": 2.0 * skill + rng.normal(0, 0.2, n),
            "target_possession_volume": 182 + 5 * skill + rng.normal(0, 0.5, n),
        }
    )

    result = research_model_feature_map(
        df,
        league="NBA",
        artifacts_dir=str(tmp_path / "artifacts"),
        feature_columns=[c for c in df.columns if c not in {"game_id", "start_time_utc", "game_date_utc", "home_win"}],
        selected_models=["glm_ridge", "two_stage"],
        approve_changes=True,
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
    )

    saved = load_model_feature_map("NBA", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    assert result.registry_updated is True
    assert "glm_ridge" in saved
    assert "two_stage" in saved
    assert "diff_form_point_margin" in saved["glm_ridge"]
    assert "elo_home_prob" in saved["glm_ridge"]
    assert len(saved["glm_ridge"]) <= 10
    assert "home_ewm_shot_volume_share" in saved["two_stage"]


def test_nba_gbdt_feature_research_prunes_tree_width(tmp_path) -> None:
    n = 260
    rng = np.random.default_rng(7)
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": pd.date_range("2025-10-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-10-01", periods=n, freq="D").date.astype(str),
            "home_win": (skill + 0.25 * noise > 0).astype(int),
            "diff_form_point_margin": 1.2 * skill + rng.normal(0, 0.1, n),
            "travel_diff": 0.2 * skill + rng.normal(0, 0.2, n),
            "rest_diff": 0.2 * skill + rng.normal(0, 0.2, n),
            "elo_home_prob": 0.55 + 0.1 * np.tanh(skill),
            "dyn_home_prob": 0.54 + 0.1 * np.tanh(skill),
            "home_ewm_points_for": 112 + 5 * skill + rng.normal(0, 1, n),
            "away_ewm_points_for": 108 - 4 * skill + rng.normal(0, 1, n),
            "home_ewm_point_margin": 6 * skill + rng.normal(0, 0.6, n),
            "away_ewm_point_margin": -5 * skill + rng.normal(0, 0.6, n),
            "target_shot_volume_share": 0.5 + 0.08 * np.tanh(skill),
            "target_free_throw_pressure": 2.0 * skill + rng.normal(0, 0.2, n),
            "target_possession_volume": 182 + 5 * skill + rng.normal(0, 0.5, n),
        }
    )

    for idx in range(60):
        df[f"tree_signal_{idx:02d}"] = (0.9 - 0.01 * idx) * skill + rng.normal(0, 0.25 + 0.01 * idx, n)

    result = research_model_feature_map(
        df,
        league="NBA",
        artifacts_dir=str(tmp_path / "artifacts"),
        feature_columns=[c for c in df.columns if c not in {"game_id", "start_time_utc", "game_date_utc", "home_win"}],
        selected_models=["gbdt"],
        approve_changes=True,
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
    )

    saved = load_model_feature_map("NBA", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    assert result.registry_updated is True
    assert "gbdt" in saved
    assert len(saved["gbdt"]) <= 40
    assert "diff_form_point_margin" in saved["gbdt"]
    assert "elo_home_prob" in saved["gbdt"]


def test_nhl_model_feature_research_promotes_per_model_feature_map(tmp_path) -> None:
    n = 260
    rng = np.random.default_rng(21)
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": pd.date_range("2025-10-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-10-01", periods=n, freq="D").date.astype(str),
            "home_win": (skill + 0.2 * noise > 0).astype(int),
            "diff_form_goal_diff": 1.1 * skill + rng.normal(0, 0.1, n),
            "diff_form_win_rate": 0.9 * skill + rng.normal(0, 0.1, n),
            "travel_diff": 0.2 * skill + rng.normal(0, 0.2, n),
            "rest_diff": 0.2 * skill + rng.normal(0, 0.2, n),
            "special_pp_diff": 0.8 * skill + rng.normal(0, 0.15, n),
            "special_pk_pressure_diff": 0.75 * skill + rng.normal(0, 0.15, n),
            "goalie_quality_diff": 0.9 * skill + rng.normal(0, 0.15, n),
            "goalie_workload_diff_7": 0.55 * skill + rng.normal(0, 0.2, n),
            "goalie_workload_diff_14": 0.45 * skill + rng.normal(0, 0.2, n),
            "goalie_uncertainty_diff": 0.4 * skill + rng.normal(0, 0.2, n),
            "diff_roster_strength": 0.7 * skill + rng.normal(0, 0.15, n),
            "diff_lineup_uncertainty": -0.35 * skill + rng.normal(0, 0.2, n),
            "diff_xg_share": 0.85 * skill + rng.normal(0, 0.15, n),
            "diff_penalty_diff": 0.65 * skill + rng.normal(0, 0.2, n),
            "elo_home_pre": 1520 + 30 * skill + rng.normal(0, 2, n),
            "elo_away_pre": 1500 - 28 * skill + rng.normal(0, 2, n),
            "elo_home_prob": 0.55 + 0.1 * np.tanh(skill),
            "dyn_home_mean": 0.6 * skill + rng.normal(0, 0.15, n),
            "dyn_away_mean": -0.55 * skill + rng.normal(0, 0.15, n),
            "dyn_home_prob": 0.54 + 0.11 * np.tanh(skill),
            "dyn_var_diff": 0.3 * np.abs(skill) + rng.normal(0, 0.1, n),
            "rink_goal_effect": 0.3 * skill + rng.normal(0, 0.15, n),
            "rink_shot_effect": 0.25 * skill + rng.normal(0, 0.15, n),
            "target_xg_share": 0.5 + 0.08 * np.tanh(skill),
            "target_penalty_diff": 1.8 * skill + rng.normal(0, 0.2, n),
            "target_pace": 60 + 2.5 * skill + rng.normal(0, 0.4, n),
        }
    )

    result = research_model_feature_map(
        df,
        league="NHL",
        artifacts_dir=str(tmp_path / "artifacts"),
        feature_columns=[c for c in df.columns if c not in {"game_id", "start_time_utc", "game_date_utc", "home_win"}],
        selected_models=["glm_ridge", "bayes_bt_state_space"],
        approve_changes=True,
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
    )

    saved = load_model_feature_map("NHL", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    assert result.registry_updated is True
    assert "glm_ridge" in saved
    assert "bayes_bt_state_space" in saved
    assert "diff_form_goal_diff" in saved["glm_ridge"]
    assert "goalie_quality_diff" in saved["glm_ridge"]
    assert "travel_diff" in saved["bayes_bt_state_space"]
    assert "elo_home_prob" in saved["bayes_bt_state_space"]


def test_feature_research_preserves_other_model_maps_on_partial_update(tmp_path) -> None:
    registry_path = tmp_path / "model_feature_map_nba.yaml"
    registry_path.write_text(
        """
version: 1
league: NBA
updated_at_utc: '2026-03-05T00:00:00+00:00'
models:
  glm_ridge:
    active_features:
    - diff_form_point_margin
    - elo_home_prob
    feature_count: 2
"""
    )

    n = 260
    rng = np.random.default_rng(11)
    skill = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": pd.date_range("2025-10-01", periods=n, freq="D").astype(str),
            "game_date_utc": pd.date_range("2025-10-01", periods=n, freq="D").date.astype(str),
            "home_win": (skill > 0).astype(int),
            "diff_form_point_margin": skill + rng.normal(0, 0.1, n),
            "travel_diff": 0.2 * skill + rng.normal(0, 0.1, n),
            "rest_diff": 0.2 * skill + rng.normal(0, 0.1, n),
            "elo_home_prob": 0.55 + 0.1 * np.tanh(skill),
            "dyn_home_prob": 0.54 + 0.1 * np.tanh(skill),
            "home_ewm_points_for": 110 + 5 * skill + rng.normal(0, 1, n),
            "away_ewm_points_for": 108 - 4 * skill + rng.normal(0, 1, n),
            "home_ewm_point_margin": 6 * skill + rng.normal(0, 0.5, n),
            "away_ewm_point_margin": -5 * skill + rng.normal(0, 0.5, n),
        }
    )
    for idx in range(45):
        df[f"tree_signal_{idx:02d}"] = (0.8 - 0.01 * idx) * skill + rng.normal(0, 0.25, n)

    research_model_feature_map(
        df,
        league="NBA",
        artifacts_dir=str(tmp_path / "artifacts"),
        feature_columns=[c for c in df.columns if c not in {"game_id", "start_time_utc", "game_date_utc", "home_win"}],
        selected_models=["gbdt"],
        approve_changes=True,
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
    )

    saved = load_model_feature_map("NBA", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    assert saved["glm_ridge"] == ["diff_form_point_margin", "elo_home_prob"]
    assert "gbdt" in saved
