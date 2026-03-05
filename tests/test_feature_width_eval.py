import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.feature_width_eval import run_feature_width_eval
from src.training.model_feature_research import load_model_feature_map


def _synthetic_nhl_rf_frame(n: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    start = pd.date_range("2025-10-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "start_time_utc": start.astype(str),
            "game_date_utc": start.date.astype(str),
            "home_win": (skill + 0.25 * noise > 0).astype(int),
        }
    )

    signal_features = [
        "home_penalty_diff",
        "home_pp_eff",
        "home_starter_known",
        "rink_goal_effect",
        "home_post_trade_deadline",
        "away_pp_eff_ewm",
        "diff_roster_strength",
        "home_pp_eff_ewm",
        "rink_shot_effect",
        "home_r5_shots_against",
        "away_r5_team_save_pct_proxy",
        "home_days_into_season",
        "away_r5_shots_for",
        "home_season_phase_late",
        "away_r14_shots_for",
        "away_games_played_prior",
        "home_r14_shots_against",
        "home_r5_shots_for",
        "away_r14_team_save_pct_proxy",
        "away_ewm_shots_against",
    ]
    noise_features = [
        "away_pp_eff",
        "home_roster_strength_index",
        "home_r14_team_save_pct_proxy",
        "away_r14_shots_against",
    ]

    for idx, col in enumerate(signal_features, start=1):
        strength = 0.95 - 0.02 * idx
        df[col] = strength * skill + rng.normal(0, 0.25, n)

    for col in noise_features:
        df[col] = rng.normal(0, 1.0, n)

    df["home_starter_known"] = (df["home_starter_known"] > df["home_starter_known"].median()).astype(int)
    df["home_post_trade_deadline"] = (pd.to_datetime(df["game_date_utc"]).dt.month >= 3).astype(int)
    df["home_season_phase_late"] = (pd.to_datetime(df["game_date_utc"]).dt.month >= 4).astype(int)
    return df


def test_feature_width_eval_runs_backtests_and_promotes_best_width(tmp_path) -> None:
    df = _synthetic_nhl_rf_frame()

    result = run_feature_width_eval(
        df,
        league="NHL",
        model_name="rf",
        artifacts_dir=str(tmp_path / "artifacts"),
        bayes_cfg={},
        n_splits=3,
        candidate_widths=[20, 24],
        path_template=str(tmp_path / "model_feature_map_{league}.yaml"),
        approve_changes=True,
    )

    saved = load_model_feature_map("NHL", path_template=str(tmp_path / "model_feature_map_{league}.yaml"))
    summary_rows = json.loads(Path(result.summary_path).read_text())

    assert result.registry_updated is True
    assert result.best_width in {20, 24}
    assert len(result.best_features) == result.best_width
    assert saved["rf"] == result.best_features
    assert len(result.summary_rows) == 2
    assert {row["feature_count"] for row in result.summary_rows} == {20, 24}
    assert summary_rows[0]["feature_count"] == result.best_width
