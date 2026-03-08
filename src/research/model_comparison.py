from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd

from src.common.config import AppConfig
from src.common.utils import ensure_dir
from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.metrics import metric_bundle, per_game_scores
from src.evaluation.validation_classification import validate_logistic_probability_model
from src.evaluation.validation_nonlinearity import assess_nonlinearity
from src.features.leakage_checks import run_leakage_checks
from src.research.candidate_models import (
    BaseCandidateModel,
    CandidateFitStats,
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    MARSHingeCandidate,
    PenalizedLogitCandidate,
    VanillaGLMBinomialCandidate,
)
from src.services.train import load_features_dataframe
from src.training.cv import time_series_splits
from src.training.feature_selection import select_feature_columns
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

CORRELATION_SCREEN_THRESHOLD = 0.999
DEFAULT_BOOTSTRAP_SAMPLES = 1000


def _safe_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return value
    if not np.isfinite(numeric):
        return None
    return numeric


def _json_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _params_json(params: dict[str, Any]) -> str:
    if not params:
        return "{}"
    parts = [f"{key}={_json_value(value)}" for key, value in sorted(params.items())]
    return "; ".join(parts)


@dataclass(slots=True)
class CandidateFeatureSets:
    raw_feature_count: int
    screened_features: list[str]
    core_features: list[str]
    gam_features: list[str]
    mars_features: list[str]
    glmm_features: list[str]
    dglm_features: list[str]
    screening_frame: pd.DataFrame
    nonlinearity_summary: dict[str, Any]
    nonlinearity_frame: pd.DataFrame
    ranking_frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    model_name: str
    display_name: str
    param_grid: list[dict[str, Any]]
    builder: Callable[[CandidateFeatureSets, dict[str, Any]], BaseCandidateModel]


@dataclass(slots=True)
class CandidateTuningResult:
    model_name: str
    display_name: str
    best_params: dict[str, Any] | None
    cv_summary: pd.DataFrame
    cv_folds: pd.DataFrame
    failure_reason: str = ""


@dataclass(slots=True)
class ComparisonRunResult:
    league: str
    report_slug: str
    report_path: Path
    validation_metrics_path: Path
    test_metrics_path: Path
    bootstrap_path: Path
    recommendation_model: str
    recommendation_display_name: str
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    bootstrap_summary: pd.DataFrame


def _time_ordered_split(
    df: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df[df["home_win"].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    n_obs = len(work)
    train_end = int(round(train_fraction * n_obs))
    valid_end = int(round((train_fraction + validation_fraction) * n_obs))
    train_end = min(max(train_end, 1), max(n_obs - 2, 1))
    valid_end = min(max(valid_end, train_end + 1), n_obs)
    train_df = work.iloc[:train_end].copy()
    validation_df = work.iloc[train_end:valid_end].copy()
    test_df = work.iloc[valid_end:].copy()
    return train_df, validation_df, test_df


def _internal_nonlinearity_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[df["home_win"].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    if len(work) < 120:
        cut = max(40, int(round(0.7 * len(work))))
    else:
        cut = int(round(0.7 * len(work)))
    cut = min(max(cut, 1), max(len(work) - 1, 1))
    return work.iloc[:cut].copy(), work.iloc[cut:].copy()


def _feature_screening(train_df: pd.DataFrame, raw_features: list[str]) -> tuple[list[str], pd.DataFrame]:
    numeric = _safe_numeric_frame(train_df, raw_features)
    medians = numeric.median(numeric_only=True).fillna(0.0)
    filled = numeric.fillna(medians.reindex(raw_features)).fillna(0.0)
    y = train_df["home_win"].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for feature in raw_features:
        series = filled[feature]
        raw_series = numeric[feature]
        n_unique = int(series.nunique(dropna=False))
        std = float(series.std(ddof=0))
        corr = np.corrcoef(series.to_numpy(dtype=float), y)[0, 1] if n_unique > 1 else float("nan")
        score = abs(float(corr)) if np.isfinite(corr) else 0.0
        rows.append(
            {
                "feature": feature,
                "missing_rate": float(raw_series.isna().mean()),
                "n_unique": n_unique,
                "std": std,
                "univariate_score": score,
                "status": "candidate",
                "reason": "",
                "retained_as": feature,
                "correlation_anchor": "",
                "correlation_abs": np.nan,
            }
        )

    report = pd.DataFrame(rows).set_index("feature")
    constant_mask = (report["n_unique"] <= 1) | (report["std"] <= 1e-12)
    report.loc[constant_mask, "status"] = "dropped"
    report.loc[constant_mask, "reason"] = "constant_or_singleton_on_fit_window"

    exact_hashes: dict[tuple[Any, ...], str] = {}
    for feature in raw_features:
        if report.loc[feature, "status"] == "dropped":
            continue
        key = tuple(np.round(filled[feature].to_numpy(dtype=float), 10))
        if key in exact_hashes:
            report.loc[feature, "status"] = "dropped"
            report.loc[feature, "reason"] = "exact_duplicate"
            report.loc[feature, "retained_as"] = exact_hashes[key]
        else:
            exact_hashes[key] = feature

    survivors = [feature for feature in raw_features if report.loc[feature, "status"] == "candidate"]
    ordered = sorted(
        survivors,
        key=lambda feature: (-float(report.loc[feature, "univariate_score"]), str(feature)),
    )
    kept: list[str] = []
    for feature in ordered:
        candidate = filled[feature].to_numpy(dtype=float)
        anchor_feature = ""
        anchor_corr = float("nan")
        for retained in kept:
            corr = np.corrcoef(candidate, filled[retained].to_numpy(dtype=float))[0, 1]
            if np.isfinite(corr) and abs(corr) >= CORRELATION_SCREEN_THRESHOLD:
                anchor_feature = retained
                anchor_corr = abs(float(corr))
                break
        if anchor_feature:
            report.loc[feature, "status"] = "dropped"
            report.loc[feature, "reason"] = "near_duplicate_correlation_cluster"
            report.loc[feature, "retained_as"] = anchor_feature
            report.loc[feature, "correlation_anchor"] = anchor_feature
            report.loc[feature, "correlation_abs"] = anchor_corr
        else:
            report.loc[feature, "status"] = "kept"
            kept.append(feature)

    report = report.reset_index().sort_values(["status", "reason", "feature"], ascending=[True, True, True])
    return kept, report


def _fit_ranking_frame(fit_df: pd.DataFrame, screened_features: list[str]) -> pd.DataFrame:
    probe = PenalizedLogitCandidate(
        model_name="probe_ridge",
        display_name="Probe Ridge",
        features=screened_features,
        penalty="l2",
        c=1.0,
    )
    probe.fit(fit_df)
    coefficients = np.asarray(probe.model.coef_[0], dtype=float) if probe.model is not None else np.zeros(len(screened_features))
    frame = pd.DataFrame(
        {
            "feature": screened_features,
            "abs_scaled_coef": np.abs(coefficients),
            "rank": np.arange(1, len(screened_features) + 1),
        }
    ).sort_values(["abs_scaled_coef", "feature"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def _select_feature_sets(fit_df: pd.DataFrame, raw_features: list[str]) -> CandidateFeatureSets:
    screened_features, screening_frame = _feature_screening(fit_df, raw_features)
    ranking_frame = _fit_ranking_frame(fit_df, screened_features)
    numeric = _safe_numeric_frame(fit_df, screened_features)
    unique_counts = {feature: int(numeric[feature].dropna().nunique()) for feature in screened_features}
    continuous_ranked = [feature for feature in ranking_frame["feature"] if unique_counts.get(feature, 0) >= 8]
    core_features = ranking_frame["feature"].head(min(18, len(ranking_frame))).tolist()

    internal_train, internal_holdout = _internal_nonlinearity_split(fit_df)
    nonlinear_pool = continuous_ranked[: min(24, len(continuous_ranked))]
    nonlinearity = assess_nonlinearity(internal_train, internal_holdout, features=nonlinear_pool)
    nonlinearity_frame = nonlinearity["feature_summary"].copy()
    flagged = nonlinearity_frame[nonlinearity_frame["status"].isin(["moderate", "strong"])].copy()
    gam_features = flagged[flagged["family_hint"] == "gam"]["feature"].head(6).tolist()
    mars_features = flagged[flagged["family_hint"] == "mars"]["feature"].head(6).tolist()

    fallback_continuous = [feature for feature in core_features if unique_counts.get(feature, 0) >= 8]
    if not gam_features:
        gam_features = fallback_continuous[: min(4, len(fallback_continuous))]
    if not mars_features:
        mars_features = fallback_continuous[: min(4, len(fallback_continuous))]
    if not gam_features:
        gam_features = screened_features[: min(4, len(screened_features))]
    if not mars_features:
        mars_features = screened_features[: min(4, len(screened_features))]

    glmm_features = core_features[: min(14, len(core_features))]
    dglm_features = [feature for feature in core_features if unique_counts.get(feature, 0) >= 3][: min(14, len(core_features))]
    if not dglm_features:
        dglm_features = core_features[: min(10, len(core_features))]

    return CandidateFeatureSets(
        raw_feature_count=len(raw_features),
        screened_features=screened_features,
        core_features=core_features,
        gam_features=gam_features,
        mars_features=mars_features,
        glmm_features=glmm_features,
        dglm_features=dglm_features,
        screening_frame=screening_frame,
        nonlinearity_summary=dict(nonlinearity["summary"]),
        nonlinearity_frame=nonlinearity_frame,
        ranking_frame=ranking_frame,
    )


def _structured_linear_features(core_features: list[str], nonlinear_features: list[str], cap: int) -> list[str]:
    return [feature for feature in core_features if feature not in nonlinear_features][:cap]


def _candidate_specs(feature_sets: CandidateFeatureSets) -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            model_name="glm_ridge",
            display_name="GLM Ridge",
            param_grid=[{"c": c} for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]],
            builder=lambda fs, params: PenalizedLogitCandidate(
                model_name="glm_ridge",
                display_name="GLM Ridge",
                features=fs.screened_features,
                penalty="l2",
                c=float(params["c"]),
                solver="lbfgs",
            ),
        ),
        CandidateSpec(
            model_name="glm_elastic_net",
            display_name="Elastic Net GLM",
            param_grid=[
                {"c": c, "l1_ratio": l1_ratio}
                for c in [0.05, 0.1, 0.25, 0.5, 1.0]
                for l1_ratio in [0.1, 0.25, 0.5, 0.75]
            ],
            builder=lambda fs, params: PenalizedLogitCandidate(
                model_name="glm_elastic_net",
                display_name="Elastic Net GLM",
                features=fs.screened_features,
                penalty="elasticnet",
                c=float(params["c"]),
                l1_ratio=float(params["l1_ratio"]),
                solver="saga",
            ),
        ),
        CandidateSpec(
            model_name="glm_lasso",
            display_name="Lasso GLM",
            param_grid=[{"c": c} for c in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]],
            builder=lambda fs, params: PenalizedLogitCandidate(
                model_name="glm_lasso",
                display_name="Lasso GLM",
                features=fs.screened_features,
                penalty="l1",
                c=float(params["c"]),
                solver="saga",
            ),
        ),
        CandidateSpec(
            model_name="glm_vanilla",
            display_name="Vanilla GLM",
            param_grid=[{}],
            builder=lambda fs, params: VanillaGLMBinomialCandidate(features=fs.screened_features),
        ),
    ]

    glmm_caps = sorted({cap for cap in [6, 10, 14] if cap <= len(feature_sets.glmm_features)}) or [len(feature_sets.glmm_features)]
    specs.append(
        CandidateSpec(
            model_name="glmm_logit",
            display_name="GLMM Logit",
            param_grid=[{"feature_cap": cap} for cap in glmm_caps],
            builder=lambda fs, params: GLMMLogitCandidate(
                fixed_features=fs.glmm_features[: int(params["feature_cap"])],
            ),
        )
    )

    dglm_caps = sorted({cap for cap in [6, 10, 14] if cap <= len(feature_sets.dglm_features)}) or [len(feature_sets.dglm_features)]
    specs.append(
        CandidateSpec(
            model_name="dglm_margin",
            display_name="DGLM Margin",
            param_grid=[{"feature_cap": cap, "iterations": iterations} for cap in dglm_caps for iterations in [1, 2]],
            builder=lambda fs, params: DGLMMarginCandidate(
                features=fs.dglm_features[: int(params["feature_cap"])],
                iterations=int(params["iterations"]),
            ),
        )
    )

    gam_caps = sorted({cap for cap in [3, 5, 6] if cap <= len(feature_sets.gam_features)}) or [len(feature_sets.gam_features)]
    specs.append(
        CandidateSpec(
            model_name="gam_spline",
            display_name="GAM Spline",
            param_grid=[{"feature_cap": cap, "n_knots": n_knots, "c": c} for cap in gam_caps for n_knots in [4, 5] for c in [0.25, 0.5, 1.0]],
            builder=lambda fs, params: GAMSplineCandidate(
                linear_features=_structured_linear_features(
                    fs.core_features,
                    fs.gam_features[: int(params["feature_cap"])],
                    cap=max(4, int(params["feature_cap"]) + 2),
                ),
                spline_features=fs.gam_features[: int(params["feature_cap"])],
                n_knots=int(params["n_knots"]),
                c=float(params["c"]),
            ),
        )
    )

    mars_caps = sorted({cap for cap in [2, 4, 6] if cap <= len(feature_sets.mars_features)}) or [len(feature_sets.mars_features)]
    specs.append(
        CandidateSpec(
            model_name="mars_hinge",
            display_name="MARS Hinge",
            param_grid=[
                {"feature_cap": cap, "knots_per_feature": knots, "interaction_degree": degree, "c": c}
                for cap in mars_caps
                for knots in [3, 4]
                for degree in [1, 2]
                for c in [0.1, 0.25, 0.5]
            ],
            builder=lambda fs, params: MARSHingeCandidate(
                linear_features=_structured_linear_features(
                    fs.core_features,
                    fs.mars_features[: int(params["feature_cap"])],
                    cap=max(4, int(params["feature_cap"]) + 2),
                ),
                hinge_features=fs.mars_features[: int(params["feature_cap"])],
                knots_per_feature=int(params["knots_per_feature"]),
                interaction_degree=int(params["interaction_degree"]),
                c=float(params["c"]),
            ),
        )
    )

    return specs


def _candidate_cv_tune(
    spec: CandidateSpec,
    *,
    fit_df: pd.DataFrame,
    feature_sets: CandidateFeatureSets,
    cv_splits: int,
) -> CandidateTuningResult:
    min_train_size = min(180, max(40, len(fit_df) // 2))
    min_train_size = min(min_train_size, max(len(fit_df) - 20, 20))
    splits = time_series_splits(fit_df, n_splits=max(2, cv_splits), min_train_size=min_train_size)
    fold_rows: list[dict[str, Any]] = []
    if not splits:
        return CandidateTuningResult(
            model_name=spec.model_name,
            display_name=spec.display_name,
            best_params=None,
            cv_summary=pd.DataFrame(),
            cv_folds=pd.DataFrame(),
            failure_reason="no_time_series_splits_available",
        )

    for params in spec.param_grid:
        params_text = _params_json(params)
        for fold_number, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = fit_df.loc[tr_idx].copy().sort_values("start_time_utc")
            va = fit_df.loc[va_idx].copy().sort_values("start_time_utc")
            if tr.empty or va.empty or va["home_win"].nunique() < 2:
                fold_rows.append(
                    {
                        "model_name": spec.model_name,
                        "display_name": spec.display_name,
                        "params": params_text,
                        "fold": int(fold_number),
                        "status": "skipped",
                        "n_train": int(len(tr)),
                        "n_valid": int(len(va)),
                        "log_loss": np.nan,
                        "brier": np.nan,
                        "accuracy": np.nan,
                        "auc": np.nan,
                        "error": "empty_or_single_class_validation_fold",
                    }
                )
                continue
            try:
                model = spec.builder(feature_sets, params)
                model.fit(tr)
                p = model.predict_proba(va)
                metrics = metric_bundle(va["home_win"].astype(int).to_numpy(), p)
                fold_rows.append(
                    {
                        "model_name": spec.model_name,
                        "display_name": spec.display_name,
                        "params": params_text,
                        "fold": int(fold_number),
                        "status": "ok",
                        "n_train": int(len(tr)),
                        "n_valid": int(len(va)),
                        "log_loss": float(metrics["log_loss"]),
                        "brier": float(metrics["brier"]),
                        "accuracy": float(metrics["accuracy"]),
                        "auc": float(metrics["auc"]),
                        "error": "",
                    }
                )
            except Exception as exc:
                fold_rows.append(
                    {
                        "model_name": spec.model_name,
                        "display_name": spec.display_name,
                        "params": params_text,
                        "fold": int(fold_number),
                        "status": "failed",
                        "n_train": int(len(tr)),
                        "n_valid": int(len(va)),
                        "log_loss": np.nan,
                        "brier": np.nan,
                        "accuracy": np.nan,
                        "auc": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    fold_frame = pd.DataFrame(fold_rows)
    if fold_frame.empty:
        return CandidateTuningResult(
            model_name=spec.model_name,
            display_name=spec.display_name,
            best_params=None,
            cv_summary=pd.DataFrame(),
            cv_folds=fold_frame,
            failure_reason="no_cv_rows_generated",
        )

    ok_folds = fold_frame[fold_frame["status"] == "ok"].copy()
    if ok_folds.empty:
        failure_reason = "; ".join(sorted(set(fold_frame["error"].dropna().astype(str).tolist())))
        return CandidateTuningResult(
            model_name=spec.model_name,
            display_name=spec.display_name,
            best_params=None,
            cv_summary=pd.DataFrame(),
            cv_folds=fold_frame,
            failure_reason=failure_reason or "all_cv_fits_failed",
        )

    summary_rows: list[dict[str, Any]] = []
    for params_text, bucket in ok_folds.groupby("params", sort=False):
        summary_rows.append(
            {
                "model_name": spec.model_name,
                "display_name": spec.display_name,
                "params": params_text,
                "folds_used": int(len(bucket)),
                "mean_log_loss": float(bucket["log_loss"].mean()),
                "mean_brier": float(bucket["brier"].mean()),
                "mean_accuracy": float(bucket["accuracy"].mean()),
                "mean_auc": float(bucket["auc"].mean()),
                "std_log_loss": float(bucket["log_loss"].std(ddof=0)),
                "std_brier": float(bucket["brier"].std(ddof=0)),
                "std_accuracy": float(bucket["accuracy"].std(ddof=0)),
                "std_auc": float(bucket["auc"].std(ddof=0)),
            }
        )
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["mean_log_loss", "mean_brier", "mean_auc", "params"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    best_params_text = str(summary_frame.iloc[0]["params"]) if not summary_frame.empty else ""
    best_params = next((params for params in spec.param_grid if _params_json(params) == best_params_text), None)
    return CandidateTuningResult(
        model_name=spec.model_name,
        display_name=spec.display_name,
        best_params=best_params,
        cv_summary=summary_frame,
        cv_folds=fold_frame,
        failure_reason="",
    )


def _diagnostic_row(
    *,
    split_name: str,
    model_name: str,
    display_name: str,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    metrics = metric_bundle(y_true, p_pred)
    calibration = calibration_alpha_beta(y_true, p_pred) | ece_mce(y_true, p_pred)
    brier = brier_decompose(y_true, p_pred)
    classification = validate_logistic_probability_model(y_true, p_pred, bins=10)
    row = {
        "split": split_name,
        "model_name": model_name,
        "display_name": display_name,
        "params": _params_json(params),
        "log_loss": float(metrics["log_loss"]),
        "brier": float(metrics["brier"]),
        "accuracy": float(metrics["accuracy"]),
        "auc": float(metrics["auc"]),
        "ece": _safe_float(calibration.get("ece")),
        "mce": _safe_float(calibration.get("mce")),
        "calibration_alpha": _safe_float(calibration.get("calibration_alpha")),
        "calibration_beta": _safe_float(calibration.get("calibration_beta")),
        "brier_reliability": _safe_float(brier.get("reliability")),
        "brier_resolution": _safe_float(brier.get("resolution")),
        "brier_uncertainty": _safe_float(brier.get("uncertainty")),
        "mean_abs_calibration_gap": _safe_float(classification["quantile_summary"].get("mean_abs_calibration_gap")),
        "max_abs_calibration_gap": _safe_float(classification["quantile_summary"].get("max_abs_calibration_gap")),
        "top_quantile_actual_lift": _safe_float(classification["lift_summary"].get("top_quantile_actual_lift")),
        "top_vs_bottom_actual_lift_ratio": _safe_float(classification["lift_summary"].get("top_vs_bottom_actual_lift_ratio")),
        "normalized_gini": _safe_float(classification["lorenz_summary"].get("normalized_gini")),
        "top_decile_event_capture": _safe_float(classification["lorenz_summary"].get("top_decile_event_capture")),
        "roc_operating_point_best_balanced_accuracy": _safe_float(
            pd.DataFrame(classification["operating_points"]).sort_values("balanced_accuracy", ascending=False).iloc[0]["threshold"]
        )
        if len(classification["operating_points"])
        else None,
        "tossup_share_current_band": _safe_float(classification["tossup_summary"].get("current_tossup_share")),
    }
    return row


def _phase_evaluation(
    *,
    split_name: str,
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    raw_features: list[str],
    cv_splits: int,
) -> tuple[CandidateFeatureSets, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = _select_feature_sets(fit_df, raw_features)
    specs = _candidate_specs(feature_sets)
    metric_rows: list[dict[str, Any]] = []
    fit_stat_rows: list[dict[str, Any]] = []
    cv_summary_frames: list[pd.DataFrame] = []
    cv_fold_frames: list[pd.DataFrame] = []

    metadata_columns = [column for column in ["game_id", "game_date_utc", "home_team", "away_team", "home_win"] if column in eval_df.columns]
    predictions = eval_df[metadata_columns].copy()
    if "game_id" not in predictions.columns:
        predictions.insert(0, "game_id", np.arange(len(predictions), dtype=int))
    y_eval = eval_df["home_win"].astype(int).to_numpy()

    intercept_probability = float(fit_df["home_win"].astype(int).mean())
    intercept_pred = np.full(len(eval_df), intercept_probability, dtype=float)
    predictions["intercept_only"] = intercept_pred
    intercept_row = _diagnostic_row(
        split_name=split_name,
        model_name="intercept_only",
        display_name="Intercept Only",
        y_true=y_eval,
        p_pred=intercept_pred,
        params={"p_home_win": intercept_probability},
    )
    intercept_row["fit_status"] = "ok"
    intercept_row["fit_error"] = ""
    metric_rows.append(intercept_row)
    fit_stat_rows.append(
        CandidateFitStats(
            model_name="intercept_only",
            display_name="Intercept Only",
            parameter_count=1,
            active_parameter_count=1,
            n_features=0,
            train_log_likelihood=float(
                np.sum(
                    fit_df["home_win"].astype(int).to_numpy() * np.log(intercept_probability)
                    + (1.0 - fit_df["home_win"].astype(int).to_numpy()) * np.log(1.0 - intercept_probability)
                )
            ),
            train_deviance=None,
            train_aic=None,
            train_bic=None,
            notes="Empirical home-win base rate on the fit window",
        ).to_row()
        | {"split": split_name, "params": _params_json({"p_home_win": intercept_probability})}
    )

    for spec in specs:
        tuning = _candidate_cv_tune(
            spec,
            fit_df=fit_df,
            feature_sets=feature_sets,
            cv_splits=cv_splits,
        )
        if not tuning.cv_summary.empty:
            cv_summary_frames.append(tuning.cv_summary.copy())
        if not tuning.cv_folds.empty:
            cv_fold_frames.append(tuning.cv_folds.copy())

        if tuning.best_params is None:
            metric_rows.append(
                {
                    "split": split_name,
                    "model_name": spec.model_name,
                    "display_name": spec.display_name,
                    "params": "",
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "ece": np.nan,
                    "mce": np.nan,
                    "calibration_alpha": np.nan,
                    "calibration_beta": np.nan,
                    "brier_reliability": np.nan,
                    "brier_resolution": np.nan,
                    "brier_uncertainty": np.nan,
                    "mean_abs_calibration_gap": np.nan,
                    "max_abs_calibration_gap": np.nan,
                    "top_quantile_actual_lift": np.nan,
                    "top_vs_bottom_actual_lift_ratio": np.nan,
                    "normalized_gini": np.nan,
                    "top_decile_event_capture": np.nan,
                    "roc_operating_point_best_balanced_accuracy": np.nan,
                    "tossup_share_current_band": np.nan,
                    "fit_status": "failed",
                    "fit_error": tuning.failure_reason,
                }
            )
            continue

        try:
            model = spec.builder(feature_sets, tuning.best_params)
            model.fit(fit_df)
            p_pred = model.predict_proba(eval_df)
            predictions[spec.model_name] = p_pred
            metric_row = _diagnostic_row(
                split_name=split_name,
                model_name=spec.model_name,
                display_name=spec.display_name,
                y_true=y_eval,
                p_pred=p_pred,
                params=tuning.best_params,
            )
            metric_row["fit_status"] = "ok"
            metric_row["fit_error"] = ""
            metric_rows.append(metric_row)
            fit_stats: CandidateFitStats = model.fit_statistics()
            fit_stat_row = fit_stats.to_row()
            fit_stat_row["split"] = split_name
            fit_stat_row["params"] = _params_json(tuning.best_params)
            fit_stat_rows.append(fit_stat_row)
        except Exception as exc:
            metric_rows.append(
                {
                    "split": split_name,
                    "model_name": spec.model_name,
                    "display_name": spec.display_name,
                    "params": _params_json(tuning.best_params),
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "ece": np.nan,
                    "mce": np.nan,
                    "calibration_alpha": np.nan,
                    "calibration_beta": np.nan,
                    "brier_reliability": np.nan,
                    "brier_resolution": np.nan,
                    "brier_uncertainty": np.nan,
                    "mean_abs_calibration_gap": np.nan,
                    "max_abs_calibration_gap": np.nan,
                    "top_quantile_actual_lift": np.nan,
                    "top_vs_bottom_actual_lift_ratio": np.nan,
                    "normalized_gini": np.nan,
                    "top_decile_event_capture": np.nan,
                    "roc_operating_point_best_balanced_accuracy": np.nan,
                    "tossup_share_current_band": np.nan,
                    "fit_status": "failed",
                    "fit_error": f"{type(exc).__name__}: {exc}",
                }
            )

    metric_frame = pd.DataFrame(metric_rows).sort_values(
        ["fit_status", "log_loss", "brier", "auc", "display_name"],
        ascending=[True, True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    fit_stat_frame = pd.DataFrame(fit_stat_rows)
    cv_summary = pd.concat(cv_summary_frames, ignore_index=True) if cv_summary_frames else pd.DataFrame()
    cv_folds = pd.concat(cv_fold_frames, ignore_index=True) if cv_fold_frames else pd.DataFrame()
    return feature_sets, metric_frame, predictions, fit_stat_frame, (cv_summary if not cv_summary.empty else cv_folds)


def _bootstrap_against_best(
    prediction_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    valid = metrics_frame[metrics_frame["fit_status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame()
    best = valid.sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False]).iloc[0]
    best_model = str(best["model_name"])
    y_true = prediction_frame["home_win"].astype(int).to_numpy()
    score_map = {
        model_name: per_game_scores(y_true, prediction_frame[model_name].to_numpy())
        for model_name in valid["model_name"].tolist()
        if model_name in prediction_frame.columns
    }
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for model_name, scores in score_map.items():
        if model_name == best_model:
            continue
        diffs_log_loss: list[float] = []
        diffs_brier: list[float] = []
        other_scores = scores.reset_index(drop=True)
        best_scores = score_map[best_model].reset_index(drop=True)
        n_obs = len(best_scores)
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, n_obs, size=n_obs)
            diffs_log_loss.append(float(other_scores.loc[sample_idx, "log_loss"].mean() - best_scores.loc[sample_idx, "log_loss"].mean()))
            diffs_brier.append(float(other_scores.loc[sample_idx, "brier"].mean() - best_scores.loc[sample_idx, "brier"].mean()))

        log_loss_array = np.asarray(diffs_log_loss, dtype=float)
        brier_array = np.asarray(diffs_brier, dtype=float)
        rows.append(
            {
                "reference_model": best_model,
                "comparison_model": model_name,
                "delta_log_loss_mean": float(log_loss_array.mean()),
                "delta_log_loss_p025": float(np.quantile(log_loss_array, 0.025)),
                "delta_log_loss_p500": float(np.quantile(log_loss_array, 0.5)),
                "delta_log_loss_p975": float(np.quantile(log_loss_array, 0.975)),
                "delta_log_loss_prob_reference_better": float(np.mean(log_loss_array > 0.0)),
                "delta_brier_mean": float(brier_array.mean()),
                "delta_brier_p025": float(np.quantile(brier_array, 0.025)),
                "delta_brier_p500": float(np.quantile(brier_array, 0.5)),
                "delta_brier_p975": float(np.quantile(brier_array, 0.975)),
                "delta_brier_prob_reference_better": float(np.mean(brier_array > 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_log_loss_mean", "delta_brier_mean"], ascending=[False, False]).reset_index(drop=True)


def _file_prefix(league: str, report_slug: str) -> str:
    return f"{report_slug}_{league.lower()}"


def _best_model_row(metrics_frame: pd.DataFrame) -> pd.Series:
    valid = metrics_frame[metrics_frame["fit_status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("No candidate models completed successfully")
    return valid.sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False]).iloc[0]


def _write_report(
    *,
    cfg: AppConfig,
    report_path: Path,
    raw_feature_count: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_features: CandidateFeatureSets,
    final_features: CandidateFeatureSets,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> tuple[str, str]:
    best_validation = _best_model_row(validation_metrics)
    best_test = _best_model_row(test_metrics)
    candidate_validation = validation_metrics[validation_metrics["model_name"] != "intercept_only"].copy()
    candidate_test = test_metrics[test_metrics["model_name"] != "intercept_only"].copy()
    best_candidate_validation = _best_model_row(candidate_validation)
    best_candidate_test = _best_model_row(candidate_test)
    winner_model = str(best_test["model_name"])
    winner_display = str(best_test["display_name"])
    screening_summary = final_features.screening_frame["status"].value_counts().to_dict()
    lines = [
        f"# {cfg.data.league} Candidate Model Comparison",
        "",
        "Protocol",
        "- Objective: maximize out-of-sample predictive accuracy for home-win probabilities.",
        "- Guidance followed from the local CAS monograph sections on train/validation/test splitting (4.3), deviance and penalized fit comparisons (6.1-6.2), residual/nonlinearity/stability checks (6.3-6.4), holdout actual-vs-predicted/lift/ROC validation (7.1-7.3), and extension candidates (10.1-10.5).",
        "- Outer split: 40% train, 30% validation, 30% final test, ordered by `start_time_utc`.",
        "- Hyperparameters were tuned with rolling time-series CV inside the fit window for each phase.",
        "- Candidate models were built from the full numeric feature pool after leakage bans, not from the repo's production `glm_feature_subset` or per-model feature map.",
        "",
        "Data",
        f"- League: {cfg.data.league}",
        f"- Historical rows used: {len(train_df) + len(validation_df) + len(test_df)}",
        f"- Train / validation / test rows: {len(train_df)} / {len(validation_df)} / {len(test_df)}",
        f"- Raw candidate features after leakage bans: {raw_feature_count}",
        f"- Final screened features retained for broad linear models: {len(final_features.screened_features)}",
        f"- Feature-screening counts on final fit window: {screening_summary}",
        "",
        "Validation Ranking",
        "```text",
        validation_metrics[
            ["display_name", "log_loss", "brier", "auc", "ece", "normalized_gini", "params", "fit_status"]
        ]
        .to_string(index=False),
        "```",
        "",
        "Final Test Ranking",
        "```text",
        test_metrics[
            ["display_name", "log_loss", "brier", "auc", "ece", "normalized_gini", "params", "fit_status"]
        ]
        .to_string(index=False),
        "```",
        "",
        "Bootstrap Against Final Winner",
        "```text",
        (bootstrap_summary.to_string(index=False) if not bootstrap_summary.empty else "No bootstrap rows were generated."),
        "```",
        "",
        "Feature Form Evidence",
        f"- Final nonlinearity headline: {final_features.nonlinearity_summary.get('headline', 'n/a')}",
        f"- Top GAM candidates: {final_features.nonlinearity_summary.get('top_gam_candidates', '') or 'none flagged'}",
        f"- Top MARS candidates: {final_features.nonlinearity_summary.get('top_mars_candidates', '') or 'none flagged'}",
        "",
        "Recommendation",
        f"- Best overall final-holdout model: {winner_display} (`{winner_model}`)",
        f"- Best named candidate on the final holdout: {best_candidate_test['display_name']} (`{best_candidate_test['model_name']}`)",
        f"- Final test log loss of the best named candidate: {float(best_candidate_test['log_loss']):.6f}",
        f"- Final test Brier score of the best named candidate: {float(best_candidate_test['brier']):.6f}",
        f"- Final test AUC of the best named candidate: {float(best_candidate_test['auc']):.6f}",
        f"- Best validation candidate: {best_candidate_validation['display_name']} (`{best_candidate_validation['model_name']}`)",
    ]
    if winner_model == "intercept_only":
        lines.extend(
            [
                "- Recommendation: do not switch to any of the tested candidates yet. None beat the intercept-only benchmark on the final holdout under proper scoring rules.",
                f"- Among the named candidates, the least-bad test model was {best_candidate_test['display_name']} (`{best_candidate_test['model_name']}`), but it still underperformed the benchmark.",
            ]
        )
    else:
        lines.append(f"- Recommendation: choose {winner_display} (`{winner_model}`) for the next round if the goal is pure out-of-sample probability accuracy among the tested options.")
    if not bootstrap_summary.empty:
        second_row = bootstrap_summary.iloc[0]
        lines.extend(
            [
                f"- Closest challenger on the test set: `{second_row['comparison_model']}`",
                f"- Mean log-loss delta versus winner (challenger minus winner): {float(second_row['delta_log_loss_mean']):.6f}",
                f"- 95% bootstrap interval for that log-loss delta: [{float(second_row['delta_log_loss_p025']):.6f}, {float(second_row['delta_log_loss_p975']):.6f}]",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n")
    return winner_model, winner_display


def run_candidate_model_comparison(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> ComparisonRunResult:
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    raw_features = select_feature_columns(features_df)
    leakage_issues = run_leakage_checks(features_df, feature_columns=raw_features)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed before model comparison: {leakage_issues}")

    historical_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    train_df, validation_df, test_df = _time_ordered_split(
        historical_df,
        train_fraction=0.4,
        validation_fraction=0.3,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
        warnings.filterwarnings("ignore", message="Laplace fitting did not converge")
        warnings.filterwarnings("ignore", message="The log link alias is deprecated")
        warnings.filterwarnings("ignore", message="overflow encountered in exp")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

        validation_features, validation_metrics, validation_predictions, validation_fit_stats, validation_cv = _phase_evaluation(
            split_name="validation",
            fit_df=train_df,
            eval_df=validation_df,
            raw_features=raw_features,
            cv_splits=max(2, int(cfg.modeling.cv_splits)),
        )
        fit_plus_validation = pd.concat([train_df, validation_df], ignore_index=True).sort_values("start_time_utc").reset_index(drop=True)
        final_features, test_metrics, test_predictions, test_fit_stats, final_cv = _phase_evaluation(
            split_name="test",
            fit_df=fit_plus_validation,
            eval_df=test_df,
            raw_features=raw_features,
            cv_splits=max(2, int(cfg.modeling.cv_splits)),
        )

    bootstrap_summary = _bootstrap_against_best(
        test_predictions,
        test_metrics,
        n_bootstrap=max(100, int(bootstrap_samples)),
        random_seed=int(cfg.modeling.random_seed),
    )

    stamp = datetime.now().strftime("%Y-%m-%d")
    slug = report_slug or f"{stamp}_candidate_model_comparison"
    history_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / "history")
    prefix = _file_prefix(cfg.data.league, slug)

    validation_metrics_path = history_dir / f"{prefix}_validation_metrics.csv"
    test_metrics_path = history_dir / f"{prefix}_test_metrics.csv"
    bootstrap_path = history_dir / f"{prefix}_bootstrap.csv"
    screening_path = history_dir / f"{prefix}_feature_screening.csv"
    nonlinearity_path = history_dir / f"{prefix}_nonlinearity.csv"
    validation_predictions_path = history_dir / f"{prefix}_validation_predictions.csv"
    test_predictions_path = history_dir / f"{prefix}_test_predictions.csv"
    fit_stats_path = history_dir / f"{prefix}_fit_stats.csv"
    cv_path = history_dir / f"{prefix}_cv_summary.csv"
    report_path = history_dir / f"{prefix}_summary.md"

    validation_metrics.to_csv(validation_metrics_path, index=False)
    test_metrics.to_csv(test_metrics_path, index=False)
    bootstrap_summary.to_csv(bootstrap_path, index=False)
    final_features.screening_frame.to_csv(screening_path, index=False)
    final_features.nonlinearity_frame.to_csv(nonlinearity_path, index=False)
    validation_predictions.to_csv(validation_predictions_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)
    pd.concat([validation_fit_stats, test_fit_stats], ignore_index=True).to_csv(fit_stats_path, index=False)

    cv_frames = []
    if isinstance(validation_cv, pd.DataFrame) and not validation_cv.empty:
        validation_cv = validation_cv.copy()
        validation_cv.insert(0, "phase", "train_to_validation")
        cv_frames.append(validation_cv)
    if isinstance(final_cv, pd.DataFrame) and not final_cv.empty:
        final_cv = final_cv.copy()
        final_cv.insert(0, "phase", "train_validation_to_test")
        cv_frames.append(final_cv)
    if cv_frames:
        pd.concat(cv_frames, ignore_index=True).to_csv(cv_path, index=False)
    else:
        pd.DataFrame().to_csv(cv_path, index=False)

    recommendation_model, recommendation_display_name = _write_report(
        cfg=cfg,
        report_path=report_path,
        raw_feature_count=len(raw_features),
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        validation_features=validation_features,
        final_features=final_features,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
    )

    return ComparisonRunResult(
        league=cfg.data.league,
        report_slug=slug,
        report_path=report_path,
        validation_metrics_path=validation_metrics_path,
        test_metrics_path=test_metrics_path,
        bootstrap_path=bootstrap_path,
        recommendation_model=recommendation_model,
        recommendation_display_name=recommendation_display_name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
    )
