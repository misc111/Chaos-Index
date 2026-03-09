import numpy as np
import pandas as pd

from src.models.glm_elastic_net import GLMElasticNetModel
from src.models.glm_lasso import GLMLassoModel
from src.models.glm_ridge import GLMRidgeModel
from src.training.train import glm_feature_subset
from src.training.tune import quick_tune_glm, quick_tune_penalized_glm


def test_glm_feature_subset_drops_identifier_features():
    cols = [
        "home_starter_goalie_id",
        "away_starter_goalie_id",
        "home_is_home",
        "away_is_home",
        "home_season",
        "away_season",
        "elo_home_prob",
        "goalie_quality_diff",
    ]
    out = glm_feature_subset(cols)
    assert "home_starter_goalie_id" not in out
    assert "away_starter_goalie_id" not in out
    assert "home_is_home" not in out
    assert "away_is_home" not in out
    assert "home_season" not in out
    assert "away_season" not in out
    assert "elo_home_prob" in out
    assert "goalie_quality_diff" in out


def test_quick_tune_glm_time_series_returns_grid_choice():
    n = 360
    x = np.linspace(-2.0, 2.0, n)
    y = (x + np.random.default_rng(42).normal(0, 0.6, n) > 0).astype(int)
    df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": x,
            "noise": np.random.default_rng(7).normal(0, 1, n),
        }
    )
    out = quick_tune_glm(df, feature_cols=["signal", "noise"], c_grid=[0.1, 0.5, 1.0, 2.0], n_splits=4, min_train_size=140)
    assert out["best_c"] in {0.1, 0.5, 1.0, 2.0}
    assert len(out["results"]) >= 1
    assert all("log_loss" in r for r in out["results"])


def test_glm_handles_missing_and_inf_values():
    df = pd.DataFrame(
        {
            "home_win": [0, 1, 0, 1, 0, 1, 1, 0],
            "f1": [0.1, 0.2, np.nan, 0.8, np.inf, 0.5, -np.inf, 0.4],
            "f2": [1.0, 0.0, 0.3, np.nan, 0.2, 0.1, 0.9, np.inf],
        }
    )
    m = GLMRidgeModel(c=0.5)
    m.fit(df, feature_columns=["f1", "f2"])
    p = m.predict_proba(df)
    assert np.isfinite(p).all()
    assert (p > 0).all()
    assert (p < 1).all()


def test_quick_tune_elastic_net_returns_l1_ratio_choice():
    n = 320
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = (1.2 * x - 0.3 * noise > 0).astype(int)
    df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": x,
            "noise": noise,
        }
    )

    out = quick_tune_penalized_glm(
        df,
        feature_cols=["signal", "noise"],
        model_name="glm_elastic_net",
        c_grid=[0.1, 0.5],
        l1_ratio_grid=[0.25, 0.75],
        n_splits=4,
        min_train_size=140,
    )

    assert out["best_c"] in {0.1, 0.5}
    assert out["best_l1_ratio"] in {0.25, 0.75}
    assert out["best_params"]["l1_ratio"] in {0.25, 0.75}
    assert len(out["results"]) >= 1


def test_quick_tune_lasso_returns_grid_choice_without_l1_ratio():
    n = 320
    rng = np.random.default_rng(15)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = (1.1 * x - 0.25 * noise > 0).astype(int)
    df = pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "signal": x,
            "noise": noise,
        }
    )

    out = quick_tune_penalized_glm(
        df,
        feature_cols=["signal", "noise"],
        model_name="glm_lasso",
        c_grid=[0.05, 0.5],
        n_splits=4,
        min_train_size=140,
    )

    assert out["best_c"] in {0.05, 0.5}
    assert out["best_l1_ratio"] is None
    assert "l1_ratio" not in out["best_params"]
    assert len(out["results"]) >= 1


def test_elastic_net_handles_missing_and_inf_values():
    df = pd.DataFrame(
        {
            "home_win": [0, 1, 0, 1, 0, 1, 1, 0],
            "f1": [0.1, 0.2, np.nan, 0.8, np.inf, 0.5, -np.inf, 0.4],
            "f2": [1.0, 0.0, 0.3, np.nan, 0.2, 0.1, 0.9, np.inf],
        }
    )
    m = GLMElasticNetModel(c=0.5, l1_ratio=0.5)
    m.fit(df, feature_columns=["f1", "f2"])
    p = m.predict_proba(df)
    assert np.isfinite(p).all()
    assert (p > 0).all()
    assert (p < 1).all()


def test_lasso_handles_missing_and_inf_values():
    df = pd.DataFrame(
        {
            "home_win": [0, 1, 0, 1, 0, 1, 1, 0],
            "f1": [0.1, 0.2, np.nan, 0.8, np.inf, 0.5, -np.inf, 0.4],
            "f2": [1.0, 0.0, 0.3, np.nan, 0.2, 0.1, 0.9, np.inf],
        }
    )
    m = GLMLassoModel(c=0.5)
    m.fit(df, feature_columns=["f1", "f2"])
    p = m.predict_proba(df)
    assert np.isfinite(p).all()
    assert (p > 0).all()
    assert (p < 1).all()
