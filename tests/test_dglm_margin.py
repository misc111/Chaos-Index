from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.evaluation.validation_pipeline import build_validation_tasks, run_validation_pipeline
from src.research.candidate_models import BaseCandidateModel, CandidateFitStats, DGLMMarginCandidate
from src.research.model_comparison import (
    CandidateFeatureSets,
    CandidateSpec,
    _build_candidate_scorecards,
    _build_dglm_margin_artifacts,
    _candidate_cv_tune,
)


def _dglm_frame(n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(20260504)
    strength = np.linspace(-2.0, 2.0, n)
    variance_signal = np.tile([0.0, 1.0, 2.0, 3.0], n // 4)
    noise = rng.normal(0.0, 0.8 + 0.35 * variance_signal, n)
    margin = np.round(1.4 * strength + noise).astype(int)
    home_score = np.maximum(0, 4 + np.maximum(margin, 0))
    away_score = np.maximum(0, 4 + np.maximum(-margin, 0))
    return pd.DataFrame(
        {
            "game_id": np.arange(n),
            "start_time_utc": pd.date_range("2026-03-01", periods=n, freq="D"),
            "strength": strength,
            "variance_signal": variance_signal,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": (home_score > away_score).astype(int),
        }
    )


def test_dglm_margin_calibrated_bridge_and_margin_diagnostics() -> None:
    df = _dglm_frame()
    model = DGLMMarginCandidate(
        features=["strength", "variance_signal"],
        dispersion_features=["variance_signal"],
        iterations=2,
        bridge="logit_calibrated",
    )

    model.fit(df)
    probabilities = model.predict_proba(df.tail(12))
    diagnostics = model.margin_diagnostics(df.tail(24))
    fit_stats = model.fit_statistics()

    assert np.all(probabilities > 0.0)
    assert np.all(probabilities < 1.0)
    assert diagnostics["status"] == "ok"
    assert diagnostics["bridge_status"] == "logit_calibrated"
    assert diagnostics["margin_rmse"] >= 0.0
    assert diagnostics["within_2sd_share"] >= diagnostics["within_1sd_share"]
    assert diagnostics["residual_diagnostics"]["margin_rmse"] == diagnostics["margin_rmse"]
    assert diagnostics["dispersion_diagnostics"]["residual_mse"] >= 0.0
    residual_frame = model.margin_prediction_frame(df.tail(24))
    assert {"residual", "predicted_sd", "standardized_residual"}.issubset(residual_frame.columns)
    assert "bridge=logit_calibrated" in fit_stats.notes
    assert "dispersion_features=1" in fit_stats.notes


def test_candidate_scorecard_carries_dglm_margin_diagnostics() -> None:
    margin_diagnostics = {
        "status": "ok",
        "bridge": "logit_calibrated",
        "bridge_status": "logit_calibrated",
        "margin_rmse": 2.1,
        "variance_to_residual_mse_ratio": 1.05,
        "residual_diagnostics": {
            "margin_rmse": 2.1,
            "mean_abs_standardized_residual": 0.82,
        },
        "dispersion_diagnostics": {
            "residual_mse": 4.41,
            "mean_predicted_variance": 4.2,
            "within_1sd_gap_vs_normal": 0.02,
        },
    }
    validation_metrics = pd.DataFrame(
        [
            {
                "model_name": "dglm_margin",
                "display_name": "DGLM Margin",
                "log_loss": 0.69,
                "brier": 0.24,
                "accuracy": 0.55,
                "auc": 0.54,
                "margin_diagnostics": margin_diagnostics,
            }
        ]
    )
    test_metrics = pd.DataFrame(
        [
            {
                "model_name": "intercept_only",
                "display_name": "Intercept Only",
                "log_loss": 0.70,
                "brier": 0.25,
                "accuracy": 0.50,
                "auc": 0.50,
            },
            {
                "model_name": "dglm_margin",
                "display_name": "DGLM Margin",
                "params": "bridge=logit_calibrated",
                "log_loss": 0.69,
                "brier": 0.24,
                "accuracy": 0.55,
                "auc": 0.54,
                "fit_status": "ok",
                "margin_diagnostics": margin_diagnostics,
            },
        ]
    )
    fit_stats = pd.DataFrame(
        [
            {
                "model_name": "dglm_margin",
                "n_features": 2,
                "active_parameter_count": 5,
            }
        ]
    )

    scorecards = _build_candidate_scorecards(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_fit_stats=fit_stats,
        bootstrap_summary=pd.DataFrame(),
        recommended_model="dglm_margin",
    )

    dglm = scorecards[0].to_dict()
    margin = dglm["complement_summary"]["margin_diagnostics"]
    assert margin["promotion_gate"] == "separate_margin_and_bridge_validation_required"
    assert margin["validation"]["margin_rmse"] == 2.1
    assert margin["final_holdout"]["bridge_status"] == "logit_calibrated"

    payload, metrics = _build_dglm_margin_artifacts(
        candidate_scorecards=scorecards,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    assert payload is not None
    assert payload["governance_contract_version"]
    assert payload["target_scope"]["market"] == "moneyline"
    assert payload["theory_classification"] == "theory-compatible extension"
    assert payload["bridge"] == "logit_calibrated"
    assert payload["chosen_params"] == "bridge=logit_calibrated"
    assert payload["promotion_gate"] == "separate_margin_and_bridge_validation_required"
    assert payload["final_holdout"]["margin_diagnostics"]["variance_to_residual_mse_ratio"] == 1.05
    assert payload["final_holdout"]["residual_diagnostics"]["mean_abs_standardized_residual"] == 0.82
    assert payload["dispersion_diagnostics"]["final_holdout"]["residual_mse"] == 4.41
    assert metrics.loc[metrics["phase"] == "final_holdout", "margin_rmse"].iloc[0] == 2.1
    assert metrics.loc[metrics["phase"] == "final_holdout", "residual_mse"].iloc[0] == 4.41


def test_validation_pipeline_writes_dglm_margin_diagnostics(tmp_path) -> None:
    cfg = load_config("configs/default.yaml")
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.data.league = "MLB"
    df = _dglm_frame(120)

    result = {
        "models": {},
        "train_df": df,
        "feature_columns": ["strength", "variance_signal"],
        "run_payload": {
            "model_run_id": "dglm_validation_unit",
            "feature_set_version": "fset_dglm_validation_unit",
            "selected_models": ["dglm_margin"],
            "glm_primary_model": "dglm_margin",
            "model_feature_columns": {"dglm_margin": ["strength", "variance_signal"]},
            "target_name": "moneyline_home_win",
            "target_col": "home_win",
        },
    }

    tasks = [task for task in build_validation_tasks() if task.name == "dglm_margin_diagnostics"]
    outputs = run_validation_pipeline(result, cfg, tasks=tasks)

    records = {record["task_name"]: record for record in outputs.task_records}
    record = records["dglm_margin_diagnostics"]
    assert record["validation_lane"] == "theory_compatible_engineering"
    assert record["theory_classification"] == "theory-compatible extension"
    assert record["summary"]["promotion_gate"] == "separate_margin_and_bridge_validation_required"
    assert record["summary"]["margin_summary"]["status"] == "ok"
    validation_root = tmp_path / "artifacts" / "validation" / "mlb"
    assert (validation_root / "diagnostics" / "dglm_margin" / "validation_dglm_margin_summary.json").exists()
    assert (validation_root / "diagnostics" / "dglm_margin" / "validation_dglm_margin_metrics.csv").exists()
    assert (validation_root / "diagnostics" / "dglm_margin" / "validation_dglm_margin_residuals.csv").exists()
    assert (validation_root / "diagnostics" / "dglm_margin" / "validation_dglm_margin_residual_diagnostics.json").exists()
    assert (validation_root / "diagnostics" / "dglm_margin" / "validation_dglm_margin_dispersion_bins.csv").exists()
    assert (validation_root / "diagnostics" / "dglm_margin" / "validation_dglm_margin_dispersion_diagnostics.json").exists()


def test_dglm_cv_dispersion_ranking_uses_fold_train_slice(monkeypatch) -> None:
    df = _dglm_frame(120)
    calls: list[int] = []

    def spy_rank(fit_df: pd.DataFrame, features: list[str]) -> list[str]:
        calls.append(len(fit_df))
        return list(reversed(features))

    class DummyDGLM(BaseCandidateModel):
        model_name = "dglm_margin"
        display_name = "DGLM Margin"

        def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
            self.n = len(df)

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            signal = pd.to_numeric(df["strength"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return np.clip(0.5 + 0.05 * signal, 0.05, 0.95)

        def fit_statistics(self) -> CandidateFitStats:
            return CandidateFitStats("dglm_margin", "DGLM Margin", 1, 1, 1)

    feature_sets = CandidateFeatureSets(
        raw_feature_count=2,
        screened_features=["strength", "variance_signal"],
        core_features=["strength", "variance_signal"],
        gam_features=[],
        mars_features=[],
        glmm_features=[],
        dglm_features=["strength", "variance_signal"],
        dglm_dispersion_features=["strength", "variance_signal"],
        screening_frame=pd.DataFrame(),
        nonlinearity_summary={},
        nonlinearity_frame=pd.DataFrame(),
        ranking_frame=pd.DataFrame(),
    )
    spec = CandidateSpec(
        model_name="dglm_margin",
        display_name="DGLM Margin",
        param_grid=[{"bridge": "normal"}],
        builder=lambda fs, params: DummyDGLM(),
    )
    monkeypatch.setattr("src.research.model_comparison._rank_dglm_dispersion_features", spy_rank)

    result = _candidate_cv_tune(spec, fit_df=df, feature_sets=feature_sets, cv_splits=2)

    assert result.best_params == {"bridge": "normal"}
    assert calls
    assert all(call < len(df) for call in calls)
