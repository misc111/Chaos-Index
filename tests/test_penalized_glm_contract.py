from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.glm_penalized import GLMLassoModel, build_penalized_glm
from src.training.contracts import PenaltySelection
from src.training.penalized_glm import (
    build_penalized_glm_penalty_selection,
    collect_penalized_glm_artifact_payloads,
)
from src.training.tune import quick_tune_penalized_glm


def _synthetic_penalized_glm_frame(n_rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    signal = rng.normal(0.0, 1.0, n_rows)
    noise = rng.normal(0.0, 1.0, n_rows)
    weak = rng.normal(0.0, 1.0, n_rows)
    linear = 1.8 * signal - 0.35 * noise + 0.05 * weak
    probability = 1.0 / (1.0 + np.exp(-linear))
    outcome = (rng.uniform(0.0, 1.0, n_rows) < probability).astype(int)
    return pd.DataFrame(
        {
            "start_time_utc": pd.date_range("2025-01-01", periods=n_rows, freq="D").astype(str),
            "home_win": outcome,
            "signal": signal,
            "noise": noise,
            "weak": weak,
        }
    )


def test_penalized_glm_model_reports_active_coefficient_metadata():
    df = _synthetic_penalized_glm_frame()

    model = GLMLassoModel(c=0.25)
    model.fit(df, feature_columns=["signal", "noise", "weak"])

    coef_frame = model.coef_frame()
    assert {
        "feature",
        "coef",
        "coef_scaled",
        "coef_original",
        "active",
        "active_rank",
        "scale",
        "median_impute_value",
        "lambda",
        "c",
        "l1_ratio",
    }.issubset(coef_frame.columns)
    assert coef_frame["abs_coef_scaled"].is_monotonic_decreasing

    fit_summary = model.fit_metadata_dict(top_n=2)
    assert fit_summary["penalty_family"] == "lasso"
    assert fit_summary["active_parameter_count"] >= 1
    assert len(fit_summary["active_coefficient_summary"]) <= 2
    assert fit_summary["active_coefficient_totals"]["active_parameter_count"] == fit_summary["active_parameter_count"]
    assert fit_summary["active_coefficient_totals"]["top_n_effective"] == len(fit_summary["active_coefficient_summary"])
    assert fit_summary["coefficient_path_metadata"]["path_columns"] == list(coef_frame.columns)
    assert fit_summary["coefficient_path_metadata"]["n_features_total"] == len(coef_frame)
    assert fit_summary["coefficient_path_metadata"]["n_features_active"] == fit_summary["active_parameter_count"]
    assert fit_summary["coefficient_path_metadata"]["top_n_effective"] <= 2
    assert len(fit_summary["coefficient_path_metadata"]["top_path_rows"]) <= 2
    assert all("abs_coef_scaled_share" in row for row in fit_summary["active_coefficient_summary"])
    assert set(fit_summary["active_features"]).issubset({"signal", "noise", "weak"})


def test_quick_tune_penalized_glm_emits_consistent_penalty_metadata():
    df = _synthetic_penalized_glm_frame()

    out = quick_tune_penalized_glm(
        df,
        feature_cols=["signal", "noise", "weak"],
        model_name="glm_elastic_net",
        c_grid=[0.25, 0.5],
        l1_ratio_grid=[0.25, 0.75],
        n_splits=4,
        min_train_size=140,
    )

    assert out["penalty_family"] == "elastic_net"
    assert out["best_penalty"]["model_name"] == "glm_elastic_net"
    assert out["best_penalty"]["lambda"] == pytest.approx(1.0 / out["best_penalty"]["c"])
    assert out["best_penalty"]["l1_ratio"] == pytest.approx(out["best_l1_ratio"])
    assert out["best_active_parameter_count"] == out["best_fit_summary"]["active_parameter_count"]
    assert out["best_active_features"] == out["best_fit_summary"]["active_features"]
    assert out["best_active_coefficient_summary"] == out["best_fit_summary"]["active_coefficient_summary"]

    required_keys = {"model_name", "penalty_family", "lambda", "c", "l1_ratio"}
    assert all(required_keys.issubset(row) for row in out["search_grid"])
    assert all(required_keys.issubset(row) for row in out["results"])
    assert all(required_keys.issubset(row) for row in out["fold_metrics"])
    assert all("active_parameter_count_mean" in row for row in out["results"])
    assert all("leading_features" in row for row in out["fold_metrics"])


def test_penalized_glm_contract_helpers_build_contract_ready_payloads():
    df = _synthetic_penalized_glm_frame()
    tuning = quick_tune_penalized_glm(
        df,
        feature_cols=["signal", "noise", "weak"],
        model_name="glm_elastic_net",
        c_grid=[0.25, 0.5],
        l1_ratio_grid=[0.25, 0.75],
        n_splits=4,
        min_train_size=140,
    )
    model = build_penalized_glm(
        "glm_elastic_net",
        c=float(tuning["best_c"]),
        l1_ratio=float(tuning["best_l1_ratio"]),
    )
    model.fit(df, feature_columns=["signal", "noise", "weak"])

    selection = build_penalized_glm_penalty_selection(
        "glm_elastic_net",
        tuning=tuning,
        model=model,
    )
    payloads = collect_penalized_glm_artifact_payloads(
        selected_models=["glm_elastic_net"],
        models={"glm_elastic_net": model},
        tuning_by_model={"glm_elastic_net": tuning},
        top_n=3,
    )

    assert isinstance(selection, PenaltySelection)
    assert selection.lambda_value == pytest.approx(tuning["best_lambda"])
    assert selection.c_value == pytest.approx(tuning["best_c"])
    assert selection.l1_ratio == pytest.approx(tuning["best_l1_ratio"])

    payload = payloads["glm_elastic_net"]
    assert isinstance(payload["penalty"], PenaltySelection)
    assert payload["penalty"].active_parameter_count == payload["fit_summary"]["active_parameter_count"]
    assert payload["penalty"].active_features == payload["fit_summary"]["active_features"]
    assert payload["coefficient_path_metadata"]["parameterization"]["c"] == pytest.approx(tuning["best_c"])
    assert payload["tuning_summary"]["evaluated_parameter_count"] == len(tuning["results"])
    assert payload["fit_summary"]["tuning_summary"]["best_parameterization"]["c"] == pytest.approx(tuning["best_c"])
    assert payload["fit_summary"]["active_coefficient_totals"]["active_parameter_count"] == payload["fit_summary"]["active_parameter_count"]


def test_penalized_glm_payloads_include_tuning_fallback_without_fitted_model():
    df = _synthetic_penalized_glm_frame()
    tuning = quick_tune_penalized_glm(
        df,
        feature_cols=["signal", "noise", "weak"],
        model_name="glm_lasso",
        c_grid=[0.25, 0.5],
        n_splits=4,
        min_train_size=140,
    )

    payloads = collect_penalized_glm_artifact_payloads(
        selected_models=["glm_lasso"],
        models={},
        tuning_by_model={"glm_lasso": tuning},
        top_n=3,
    )

    payload = payloads["glm_lasso"]
    fit_summary = payload["fit_summary"]

    assert isinstance(payload["penalty"], PenaltySelection)
    assert fit_summary["active_parameter_count"] == tuning["best_active_parameter_count"]
    assert fit_summary["active_features"] == tuning["best_active_features"]
    assert fit_summary["coefficient_path_metadata"]["top_n_effective"] <= 3
    assert fit_summary["coefficient_path_metadata"]["parameterization"]["c"] == pytest.approx(tuning["best_c"])
    assert fit_summary["tuning_summary"]["selected_penalty_rank"] is not None
    assert fit_summary["tuning_summary"]["top_candidates"]
