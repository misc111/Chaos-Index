import numpy as np
import pandas as pd

from src.training.model_feature_research import load_model_feature_map, research_model_feature_map


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
        selected_models=["glm_logit", "two_stage"],
        approve_changes=True,
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
    )

    saved = load_model_feature_map("NBA", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    assert result.registry_updated is True
    assert "glm_logit" in saved
    assert "two_stage" in saved
    assert "diff_form_point_margin" in saved["glm_logit"]
    assert "discipline_free_throw_pressure_diff" in saved["glm_logit"]
    assert "home_ewm_shot_volume_share" in saved["two_stage"]
