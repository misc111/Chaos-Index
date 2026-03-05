import json

import numpy as np
import pandas as pd
import pytest

from src.training.train import normalize_selected_models, train_and_predict


def _synthetic_features(n_train: int = 260, n_upcoming: int = 20) -> pd.DataFrame:
    n = n_train + n_upcoming
    rng = np.random.default_rng(123)

    start = pd.date_range("2025-01-01", periods=n, freq="D")
    base = pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [20252026] * n,
            "game_date_utc": start.date.astype(str),
            "start_time_utc": start.astype(str),
            "home_team": ["TOR"] * n,
            "away_team": ["MTL"] * n,
            "venue": ["X"] * n,
            "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
            "status_final": [1] * n_train + [0] * n_upcoming,
        }
    )

    # Core signal for synthetic outcome generation.
    skill = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)

    feature_names = [
        "diff_form_goal_diff",
        "diff_form_win_rate",
        "diff_xg_share",
        "diff_penalty_diff",
        "diff_roster_strength",
        "diff_lineup_uncertainty",
        "special_pp_diff",
        "special_penalty_diff",
        "special_pk_pressure_diff",
        "goalie_quality_diff",
        "goalie_workload_diff_7",
        "goalie_workload_diff_14",
        "goalie_uncertainty_diff",
        "travel_diff",
        "rest_diff",
        "rink_goal_effect",
        "rink_shot_effect",
        "elo_home_pre",
        "elo_away_pre",
        "elo_home_prob",
        "dyn_home_mean",
        "dyn_away_mean",
        "dyn_home_prob",
        "dyn_var_diff",
    ]
    for i, col in enumerate(feature_names, start=1):
        base[col] = 0.2 * skill + 0.05 * i + 0.3 * noise + rng.normal(0, 0.2, n)

    p = 1 / (1 + np.exp(-(0.8 * skill + 0.2 * noise)))
    y = (rng.uniform(0, 1, n_train) < p[:n_train]).astype(float)
    base["home_win"] = list(y) + [np.nan] * n_upcoming
    base["home_score"] = [3 if v == 1 else 2 for v in y] + [np.nan] * n_upcoming
    base["away_score"] = [2 if v == 1 else 3 for v in y] + [np.nan] * n_upcoming
    return base


def test_normalize_selected_models_aliases_and_validation():
    assert normalize_selected_models(["glm"]) == ["glm_logit"]
    assert "glm_logit" in normalize_selected_models(["all"])
    with pytest.raises(ValueError):
        normalize_selected_models(["not_a_model"])


def test_train_single_model_glm_only(tmp_path):
    df = _synthetic_features()
    out = train_and_predict(
        features_df=df,
        feature_set_version="test_feature_set",
        artifacts_dir=str(tmp_path / "artifacts"),
        bayes_cfg={},
        selected_models=["glm_logit"],
    )

    pred_cols = list(out["upcoming_model_probs"].columns)
    assert pred_cols == ["game_id", "glm_logit"]
    assert out["run_payload"]["selected_models"] == ["glm_logit"]
    assert out["run_payload"]["glm_best_c"] in {0.1, 0.25, 0.5, 1.0, 2.0, 4.0}

    row = out["forecasts"].iloc[0]
    per_model = json.loads(row["per_model_probs_json"])
    assert list(per_model.keys()) == ["glm_logit"]
    assert 0.0 < float(row["ensemble_prob_home_win"]) < 1.0


def test_shadow_models_stay_visible_but_out_of_ensemble(tmp_path):
    df = _synthetic_features()
    out = train_and_predict(
        features_df=df,
        feature_set_version="test_feature_set",
        artifacts_dir=str(tmp_path / "artifacts"),
        bayes_cfg={},
        selected_models=["elo_baseline", "dynamic_rating", "glm_logit", "rf"],
    )

    assert out["run_payload"]["stack_base_columns"] == ["elo_baseline", "dynamic_rating", "glm_logit"]
    assert set(out["weights"].keys()) == {"elo_baseline", "dynamic_rating", "glm_logit"}
    assert "rf" not in out["weights"]

    row = out["forecasts"].iloc[0]
    per_model = json.loads(row["per_model_probs_json"])
    assert set(per_model.keys()) == {"elo_baseline", "dynamic_rating", "glm_logit", "rf"}
