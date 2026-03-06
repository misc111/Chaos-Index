import numpy as np
import pandas as pd

from src.evaluation.validation_nonlinearity import assess_nonlinearity


def test_nonlinearity_assessment_flags_smooth_signal():
    rng = np.random.default_rng(42)
    n_train = 800
    n_holdout = 400

    x_train = rng.uniform(-2.5, 2.5, n_train)
    z_train = rng.normal(0, 1, n_train)
    p_train = 1 / (1 + np.exp(-(-1.0 + 1.6 * (x_train**2) + 0.2 * z_train)))
    y_train = (rng.uniform(0, 1, n_train) < p_train).astype(int)

    x_holdout = rng.uniform(-2.5, 2.5, n_holdout)
    z_holdout = rng.normal(0, 1, n_holdout)
    p_holdout = 1 / (1 + np.exp(-(-1.0 + 1.6 * (x_holdout**2) + 0.2 * z_holdout)))
    y_holdout = (rng.uniform(0, 1, n_holdout) < p_holdout).astype(int)

    train_df = pd.DataFrame({"home_win": y_train, "curved_signal": x_train, "noise": z_train})
    holdout_df = pd.DataFrame({"home_win": y_holdout, "curved_signal": x_holdout, "noise": z_holdout})

    report = assess_nonlinearity(train_df, holdout_df, features=["curved_signal", "noise"])
    feature_summary = report["feature_summary"].set_index("feature")

    assert report["summary"]["n_features_flagged"] >= 1
    assert feature_summary.loc["curved_signal", "status"] in {"moderate", "strong"}
    assert feature_summary.loc["curved_signal", "family_hint"] == "gam"
    assert feature_summary.loc["curved_signal", "recommendation"] in {
        "consider_gam",
        "consider_spline_transform_or_gam",
    }
    assert float(feature_summary.loc["curved_signal", "holdout_log_loss_gain_best"]) > 0


def test_nonlinearity_assessment_skips_low_cardinality_feature():
    rng = np.random.default_rng(7)
    n_train = 300
    n_holdout = 120

    train_df = pd.DataFrame(
        {
            "home_win": rng.integers(0, 2, n_train),
            "binary_feature": rng.integers(0, 2, n_train),
            "continuous_feature": rng.normal(0, 1, n_train),
        }
    )
    holdout_df = pd.DataFrame(
        {
            "home_win": rng.integers(0, 2, n_holdout),
            "binary_feature": rng.integers(0, 2, n_holdout),
            "continuous_feature": rng.normal(0, 1, n_holdout),
        }
    )

    report = assess_nonlinearity(train_df, holdout_df, features=["binary_feature", "continuous_feature"])
    feature_summary = report["feature_summary"].set_index("feature")

    assert feature_summary.loc["binary_feature", "status"] == "skip"
    assert feature_summary.loc["binary_feature", "skip_reason"] == "too_few_unique_values"
