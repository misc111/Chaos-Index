import numpy as np
import pandas as pd
import pytest

from src.training.lasso_credibility import (
    build_lasso_credibility_contracts,
    quick_tune_lasso_credibility,
    resolve_lasso_credibility_model_name,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _synthetic_mlb_frame(n: int = 320, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, n)
    discipline = rng.normal(0.0, 1.0, n)
    bullpen = rng.normal(0.0, 1.0, n)
    prior_logit = 0.55 * signal + rng.normal(0.0, 0.45, n)
    market_logit = prior_logit + 0.20 * discipline + rng.normal(0.0, 0.20, n)
    residual = 0.85 * discipline - 0.70 * bullpen
    home_win_prob = _sigmoid(prior_logit + residual)
    home_win = (rng.uniform(0.0, 1.0, n) < home_win_prob).astype(int)
    return pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-03-01", periods=n, freq="D").astype(str),
            "home_win": home_win,
            "signal_gap": signal,
            "discipline_gap": discipline,
            "bullpen_gap": bullpen,
            "prior_model_offset_logit": prior_logit,
            "market_offset_logit": market_logit,
        }
    )


def test_quick_tune_prior_offset_lasso_returns_best_lambda() -> None:
    df = _synthetic_mlb_frame()

    out = quick_tune_lasso_credibility(
        df,
        feature_cols=["signal_gap", "discipline_gap", "bullpen_gap"],
        complement_kind="prior-offset",
        complement_input_scale="logit",
        lambda_grid=[4.0, 1.0, 0.25],
        n_splits=4,
        min_train_size=140,
    )

    assert out["model_name"] == resolve_lasso_credibility_model_name("prior")
    assert out["best_lambda"] in {4.0, 1.0, 0.25}
    assert out["best_c"] == pytest.approx(1.0 / out["best_lambda"])
    assert out["best_l1_ratio"] is None
    assert len(out["results"]) >= 1
    assert all(row["complement_kind"] == "prior" for row in out["results"])


def test_build_market_offset_lasso_contracts_emits_complement_metadata() -> None:
    df = _synthetic_mlb_frame()
    tuning = quick_tune_lasso_credibility(
        df,
        feature_cols=["signal_gap", "discipline_gap", "bullpen_gap"],
        complement_kind="market",
        complement_input_scale="logit",
        lambda_grid=[2.0, 0.5, 0.125],
        n_splits=4,
        min_train_size=140,
    )

    model, artifact, scorecard = build_lasso_credibility_contracts(
        df,
        feature_cols=["signal_gap", "discipline_gap", "bullpen_gap"],
        complement_kind="market",
        complement_input_scale="logit",
        tuning=tuning,
    )

    assert artifact.model_name == "glm_lasso_market_credibility"
    assert artifact.model_family == "credibility"
    assert artifact.lane == "core"
    assert artifact.penalty is not None
    assert artifact.penalty.lambda_value == pytest.approx(tuning["best_lambda"])
    assert artifact.credibility is not None
    assert artifact.credibility.complement_kind == "market"
    assert artifact.credibility.complement_column == "market_offset_logit"
    assert artifact.credibility.offset_scale == "logit"
    assert artifact.credibility.p_values_reported is False
    assert artifact.credibility.lambda_choice_note
    assert artifact.credibility.complement_summary["offset_count"] == len(df)
    assert artifact.credibility.relativity_summary["driver_feature_count"] == 3
    assert artifact.credibility.exposure_summary["mode"] == "unit_weight"
    assert artifact.fit_summary["fit_variant"] == "market_offset_lasso_credibility"
    assert artifact.fit_summary["p_values_reported"] is False
    assert "p_value" not in artifact.fit_summary
    assert scorecard.model_name == artifact.model_name
    assert scorecard.complement_summary["offset_count"] == len(df)
    assert scorecard.feature_count == 3

    probe = pd.DataFrame(
        {
            "signal_gap": [0.0, 0.0],
            "discipline_gap": [0.1, 0.1],
            "bullpen_gap": [-0.2, -0.2],
            "market_offset_logit": [-0.8, 0.8],
        }
    )
    pred = model.predict_proba(probe)
    assert pred[1] > pred[0]


def test_lasso_credibility_fails_clearly_when_complement_input_is_missing() -> None:
    df = _synthetic_mlb_frame().drop(columns=["prior_model_offset_logit"])

    with pytest.raises(ValueError, match="prior complement column 'prior_model_offset_logit'"):
        build_lasso_credibility_contracts(
            df,
            feature_cols=["signal_gap", "discipline_gap", "bullpen_gap"],
            complement_kind="prior",
            complement_input_scale="logit",
            lambda_value=0.5,
        )
