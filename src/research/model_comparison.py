from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd

from src.common.config import AppConfig
from src.common.research import resolve_research_paths
from src.common.utils import ensure_dir, to_json
from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.metrics import metric_bundle, per_game_scores
from src.evaluation.validation_classification import validate_logistic_probability_model
from src.evaluation.validation_nonlinearity import assess_nonlinearity
from src.features.leakage_checks import run_leakage_checks
from src.registry.models import get_model_registry_entry
from src.research.artifact_guardrails import require_mlb_report_path
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
from src.research.structured_glm_specs import StructuredGLMExperimentResolution, resolve_structured_glm_experiment
from src.services.train import load_features_dataframe
from src.training.cv import time_series_splits
from src.training.contracts import CandidateComparisonContract, CandidateScorecardRecord, PromotionDecisionRecord
from src.training.feature_selection import select_feature_columns
from src.training.lambda_search import penalized_glm_search_grid
from src.training.model_feature_research import load_model_feature_map
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

CORRELATION_SCREEN_THRESHOLD = 0.999
DEFAULT_BOOTSTRAP_SAMPLES = 1000
CANDIDATE_MODEL_NAMES = {
    "glm_ridge",
    "glm_elastic_net",
    "glm_lasso",
    "glm_vanilla",
    "glmm_logit",
    "dglm_margin",
    "gam_spline",
    "mars_hinge",
}
FEATURE_POOL_FULL_SCREENED = "full_screened"
FEATURE_POOL_PRODUCTION_MODEL_MAP = "production_model_map"
FEATURE_POOL_RESEARCH_BROAD = "research_broad"
COMPARISON_TARGET_NAME = "moneyline_home_win"
COMPARISON_DISTRIBUTION = "binomial"
COMPARISON_LINK_FUNCTION = "logit"


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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = int(value)
    except Exception:
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


def _model_contract_metadata(model_name: str) -> dict[str, str]:
    try:
        entry = get_model_registry_entry(model_name)
    except KeyError:
        return {
            "lane": "benchmark" if model_name == "intercept_only" else "unknown",
            "family": "benchmark" if model_name == "intercept_only" else "unknown",
            "display_name": "Intercept Only" if model_name == "intercept_only" else model_name,
            "governance_note": "Benchmark-only reference row." if model_name == "intercept_only" else "",
        }
    return {
        "lane": entry.lane,
        "family": entry.family,
        "display_name": entry.display_label,
        "governance_note": entry.governance_note,
    }


def _is_fixture_or_demo_execution(execution_metadata: dict[str, Any] | None) -> bool:
    metadata = dict(execution_metadata or {})
    label_class = str(metadata.get("execution_label_class") or "").lower()
    scope = str(metadata.get("execution_data_scope") or metadata.get("data_scope") or "").lower()
    source = str(metadata.get("execution_source_label") or metadata.get("source_label") or "").lower()
    data_origin = str(metadata.get("data_origin") or "").lower()
    combined = f"{scope} {source} {data_origin}"
    return (
        label_class == "fixture_or_demo"
        or metadata.get("production_grade") is False
        or "fixture" in combined
        or "demo" in combined
    )


def _row_metric_slice(row: pd.Series | None, columns: list[str]) -> dict[str, Any]:
    if row is None:
        return {}
    payload: dict[str, Any] = {}
    for column in columns:
        value = row.get(column)
        if isinstance(value, (np.floating, float)):
            payload[column] = _safe_float(value)
        elif isinstance(value, (np.integer, int)):
            payload[column] = _safe_int(value)
        else:
            payload[column] = value
    return payload


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
    summary_path: Path
    candidate_scorecards_path: Path
    candidate_scorecards_contract_path: Path
    recommendation_path: Path
    validation_metrics_path: Path
    test_metrics_path: Path
    bootstrap_path: Path
    fit_stats_path: Path
    cv_path: Path
    recommendation_model: str
    recommendation_display_name: str
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    scorecards_path: Path | None = None
    leaderboard_path: Path | None = None
    leaderboard_json_path: Path | None = None
    recommendation_surface_path: Path | None = None
    artifact_manifest_path: Path | None = None
    service_output_path: Path | None = None


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


def _date_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    for column in ("start_time_utc", "game_date_utc"):
        if column in df.columns:
            values = df[column].dropna().astype(str)
            if not values.empty:
                return str(values.min()), str(values.max())
    return None, None


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


def _candidate_specs(
    feature_sets: CandidateFeatureSets,
    *,
    selected_models: set[str] | None = None,
    glm_feature_overrides: dict[str, list[str]] | None = None,
) -> list[CandidateSpec]:
    def _glm_features_for(fs: CandidateFeatureSets, model_name: str) -> list[str]:
        if not glm_feature_overrides:
            return fs.screened_features
        requested = [str(feature) for feature in glm_feature_overrides.get(model_name, []) if str(feature).strip()]
        if not requested:
            return fs.screened_features
        screened_set = set(fs.screened_features)
        resolved = [feature for feature in requested if feature in screened_set]
        return resolved or fs.screened_features

    specs = [
        CandidateSpec(
            model_name="glm_ridge",
            display_name="GLM Ridge",
            param_grid=penalized_glm_search_grid("glm_ridge"),
            builder=lambda fs, params: PenalizedLogitCandidate(
                model_name="glm_ridge",
                display_name="GLM Ridge",
                features=_glm_features_for(fs, "glm_ridge"),
                penalty="l2",
                c=float(params["c"]),
                solver="lbfgs",
            ),
        ),
        CandidateSpec(
            model_name="glm_elastic_net",
            display_name="Elastic Net GLM",
            param_grid=penalized_glm_search_grid("glm_elastic_net"),
            builder=lambda fs, params: PenalizedLogitCandidate(
                model_name="glm_elastic_net",
                display_name="Elastic Net GLM",
                features=_glm_features_for(fs, "glm_elastic_net"),
                penalty="elasticnet",
                c=float(params["c"]),
                l1_ratio=float(params["l1_ratio"]),
                solver="saga",
            ),
        ),
        CandidateSpec(
            model_name="glm_lasso",
            display_name="Lasso GLM",
            param_grid=penalized_glm_search_grid("glm_lasso"),
            builder=lambda fs, params: PenalizedLogitCandidate(
                model_name="glm_lasso",
                display_name="Lasso GLM",
                features=_glm_features_for(fs, "glm_lasso"),
                penalty="l1",
                c=float(params["c"]),
                solver="saga",
            ),
        ),
        CandidateSpec(
            model_name="glm_vanilla",
            display_name="Vanilla GLM",
            param_grid=[{}],
            builder=lambda fs, params: VanillaGLMBinomialCandidate(features=_glm_features_for(fs, "glm_vanilla")),
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

    if not selected_models:
        return specs
    return [spec for spec in specs if spec.model_name in selected_models]


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
    candidate_models: set[str] | None,
    structured_glm_resolution: StructuredGLMExperimentResolution | None = None,
) -> tuple[CandidateFeatureSets, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = _select_feature_sets(fit_df, raw_features)
    if structured_glm_resolution:
        specs = structured_glm_resolution.build_candidate_specs(
            feature_sets=feature_sets,
            selected_models=candidate_models,
            candidate_spec_builder=_candidate_specs,
        )
    else:
        specs = _candidate_specs(feature_sets, selected_models=candidate_models)
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


def _resolve_comparison_report_dirs(cfg: AppConfig, *, league: str) -> Path:
    reports_root = ensure_dir(Path(cfg.paths.artifacts_dir) / "reports")
    if str(league).upper() == "MLB":
        report_dir = ensure_dir(reports_root / "mlb")
        return require_mlb_report_path(report_dir, cfg.paths.artifacts_dir, purpose="MLB candidate comparison report")
    return ensure_dir(reports_root / "history")


def _best_model_row(metrics_frame: pd.DataFrame) -> pd.Series:
    valid = metrics_frame[metrics_frame["fit_status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("No candidate models completed successfully")
    return valid.sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False]).iloc[0]


def _candidate_metric_rows(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics_frame[metrics_frame["model_name"] != "intercept_only"].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False]).reset_index(drop=True)


def _candidate_rank_lookup(metrics_frame: pd.DataFrame) -> dict[str, int]:
    ranked = _candidate_metric_rows(metrics_frame)
    return {str(row["model_name"]): int(rank) for rank, (_, row) in enumerate(ranked.iterrows(), start=1)}


def _metric_improvement(
    *,
    model_row: pd.Series | None,
    reference_row: pd.Series | None,
    key: str,
    higher_is_better: bool = False,
) -> float | None:
    if model_row is None or reference_row is None:
        return None
    model_value = _safe_float(model_row.get(key))
    reference_value = _safe_float(reference_row.get(key))
    if model_value is None or reference_value is None:
        return None
    if higher_is_better:
        return model_value - reference_value
    return reference_value - model_value


def _metric_gap(
    *,
    model_row: pd.Series | None,
    anchor_row: pd.Series | None,
    key: str,
) -> float | None:
    if model_row is None or anchor_row is None:
        return None
    model_value = _safe_float(model_row.get(key))
    anchor_value = _safe_float(anchor_row.get(key))
    if model_value is None or anchor_value is None:
        return None
    return model_value - anchor_value


def _recommendation_tier(
    *,
    model_name: str,
    recommended_model: str,
    test_rank: int | None,
    log_loss_gap_vs_recommended: float | None,
    brier_gap_vs_recommended: float | None,
) -> str:
    if recommended_model != "intercept_only" and model_name == recommended_model:
        return "research_screen_leader"
    if test_rank is not None and test_rank <= 3:
        if (log_loss_gap_vs_recommended is None or log_loss_gap_vs_recommended <= 0.01) and (
            brier_gap_vs_recommended is None or brier_gap_vs_recommended <= 0.01
        ):
            return "challenge_watchlist"
    return "hold"


def _scorecard_badges(
    *,
    model_name: str,
    recommended_model: str,
    validation_rank: int | None,
    test_rank: int | None,
    log_loss_improvement_vs_intercept: float | None,
    auc: float | None,
) -> list[str]:
    badges: list[str] = []
    if recommended_model != "intercept_only" and model_name == recommended_model:
        badges.append("research_screen_leader")
    if test_rank == 1:
        badges.append("top_final_holdout")
    if validation_rank == 1:
        badges.append("top_validation")
    if log_loss_improvement_vs_intercept is not None and log_loss_improvement_vs_intercept > 0.0:
        badges.append("beats_intercept_log_loss")
    if auc is not None and auc >= 0.55:
        badges.append("auc_above_055")
    return badges


def _fit_stat_lookup(fit_stats_frame: pd.DataFrame) -> dict[str, pd.Series]:
    if fit_stats_frame.empty:
        return {}
    lookup: dict[str, pd.Series] = {}
    for _, row in fit_stats_frame.iterrows():
        lookup[str(row["model_name"])] = row
    return lookup


def _comparison_rejection_reasons(
    *,
    model_row: pd.Series,
    recommended_model: str,
    recommended_row: pd.Series | None,
    benchmark_row: pd.Series | None,
    bootstrap_row: pd.Series | None,
    best_named_candidate: str,
) -> list[str]:
    model_name = str(model_row["model_name"])
    if recommended_model != "intercept_only" and model_name == recommended_model:
        return []

    reasons: list[str] = []
    model_log_loss = _safe_float(model_row.get("log_loss"))
    model_brier = _safe_float(model_row.get("brier"))

    if recommended_model == "intercept_only":
        benchmark_log_loss = _safe_float(benchmark_row.get("log_loss")) if benchmark_row is not None else None
        benchmark_brier = _safe_float(benchmark_row.get("brier")) if benchmark_row is not None else None
        if model_log_loss is not None and benchmark_log_loss is not None:
            reasons.append(
                f"Final-holdout log loss {model_log_loss:.4f} trailed intercept-only at {benchmark_log_loss:.4f}."
            )
        if model_brier is not None and benchmark_brier is not None:
            reasons.append(f"Final-holdout Brier {model_brier:.4f} trailed intercept-only at {benchmark_brier:.4f}.")
        if model_name != best_named_candidate:
            reasons.append(f"Also trailed the best named candidate `{best_named_candidate}` on the final holdout.")
        return reasons[:3]

    recommended_log_loss = _safe_float(recommended_row.get("log_loss")) if recommended_row is not None else None
    recommended_brier = _safe_float(recommended_row.get("brier")) if recommended_row is not None else None
    if model_log_loss is not None and recommended_log_loss is not None and model_log_loss > recommended_log_loss + 1e-12:
        reasons.append(
            f"Final-holdout log loss {model_log_loss:.4f} trailed `{recommended_model}` at {recommended_log_loss:.4f}."
        )
    if model_brier is not None and recommended_brier is not None and model_brier > recommended_brier + 1e-12:
        reasons.append(
            f"Final-holdout Brier {model_brier:.4f} trailed `{recommended_model}` at {recommended_brier:.4f}."
        )
    if bootstrap_row is not None:
        prob = _safe_float(bootstrap_row.get("delta_log_loss_prob_reference_better"))
        if prob is not None:
            reasons.append(f"Bootstrap favored `{recommended_model}` on log loss in {prob:.1%} of resamples.")
    if not reasons:
        reasons.append(f"`{model_name}` was not selected over `{recommended_model}` on the final-holdout ranking.")
    return reasons[:3]


def _build_candidate_scorecards(
    *,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_fit_stats: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    recommended_model: str,
    execution_metadata: dict[str, Any] | None = None,
) -> list[CandidateScorecardRecord]:
    validation_lookup = {str(row["model_name"]): row for _, row in validation_metrics.iterrows()}
    validation_rank_lookup = _candidate_rank_lookup(validation_metrics)
    test_rank_lookup = _candidate_rank_lookup(test_metrics)
    test_candidates = _candidate_metric_rows(test_metrics)
    test_lookup = {str(row["model_name"]): row for _, row in test_metrics.iterrows()}
    fit_lookup = _fit_stat_lookup(test_fit_stats)
    bootstrap_lookup = {str(row["comparison_model"]): row for _, row in bootstrap_summary.iterrows()} if not bootstrap_summary.empty else {}
    benchmark_row = test_lookup.get("intercept_only")
    recommended_row = test_lookup.get(recommended_model)
    best_named_candidate = str(test_candidates.iloc[0]["model_name"]) if not test_candidates.empty else recommended_model
    fixture_or_demo = _is_fixture_or_demo_execution(execution_metadata)

    scorecards: list[CandidateScorecardRecord] = []
    for rank, (_, test_row) in enumerate(test_candidates.iterrows(), start=1):
        model_name = str(test_row["model_name"])
        metadata = _model_contract_metadata(model_name)
        validation_row = validation_lookup.get(model_name)
        fit_row = fit_lookup.get(model_name)
        bootstrap_row = bootstrap_lookup.get(model_name)
        validation_rank = validation_rank_lookup.get(model_name)
        test_rank = test_rank_lookup.get(model_name, rank)
        log_loss_improvement_vs_intercept = _metric_improvement(
            model_row=test_row,
            reference_row=benchmark_row,
            key="log_loss",
        )
        brier_improvement_vs_intercept = _metric_improvement(
            model_row=test_row,
            reference_row=benchmark_row,
            key="brier",
        )
        auc_improvement_vs_intercept = _metric_improvement(
            model_row=test_row,
            reference_row=benchmark_row,
            key="auc",
            higher_is_better=True,
        )
        log_loss_gap_vs_recommended = _metric_gap(model_row=test_row, anchor_row=recommended_row, key="log_loss")
        brier_gap_vs_recommended = _metric_gap(model_row=test_row, anchor_row=recommended_row, key="brier")
        auc_gap_vs_recommended = _metric_gap(model_row=test_row, anchor_row=recommended_row, key="auc")
        recommendation_tier = _recommendation_tier(
            model_name=model_name,
            recommended_model=recommended_model,
            test_rank=test_rank,
            log_loss_gap_vs_recommended=log_loss_gap_vs_recommended,
            brier_gap_vs_recommended=brier_gap_vs_recommended,
        )
        recommended_for_next_stage = recommended_model != "intercept_only" and model_name == recommended_model
        if fixture_or_demo and recommendation_tier == "research_screen_leader":
            recommendation_tier = "fixture_slice_screen_leader"
            recommended_for_next_stage = False
        badges = _scorecard_badges(
            model_name=model_name,
            recommended_model=recommended_model,
            validation_rank=validation_rank,
            test_rank=test_rank,
            log_loss_improvement_vs_intercept=log_loss_improvement_vs_intercept,
            auc=_safe_float(test_row.get("auc")),
        )
        if fixture_or_demo:
            badges = ["fixture_slice_screen_leader" if badge == "research_screen_leader" else badge for badge in badges]
        feature_count = _safe_int(fit_row.get("n_features")) if fit_row is not None else None
        active_parameter_count = _safe_int(fit_row.get("active_parameter_count")) if fit_row is not None else None
        active_parameter_ratio = (
            float(active_parameter_count) / float(feature_count)
            if feature_count is not None and feature_count > 0 and active_parameter_count is not None
            else None
        )
        rejection_reasons = _comparison_rejection_reasons(
            model_row=test_row,
            recommended_model=recommended_model,
            recommended_row=recommended_row,
            benchmark_row=benchmark_row,
            bootstrap_row=bootstrap_row,
            best_named_candidate=best_named_candidate,
        )
        scorecards.append(
            CandidateScorecardRecord(
                model_name=model_name,
                lane=metadata["lane"],
                target_name=COMPARISON_TARGET_NAME,
                distribution=COMPARISON_DISTRIBUTION,
                link_function=COMPARISON_LINK_FUNCTION,
                feature_count=feature_count,
                active_parameter_count=active_parameter_count,
                validation_metrics={
                    "validation_log_loss": _safe_float(validation_row.get("log_loss")) if validation_row is not None else None,
                    "validation_brier": _safe_float(validation_row.get("brier")) if validation_row is not None else None,
                    "validation_auc": _safe_float(validation_row.get("auc")) if validation_row is not None else None,
                    "validation_accuracy": _safe_float(validation_row.get("accuracy")) if validation_row is not None else None,
                    "final_holdout_log_loss": _safe_float(test_row.get("log_loss")),
                    "final_holdout_brier": _safe_float(test_row.get("brier")),
                    "final_holdout_auc": _safe_float(test_row.get("auc")),
                    "final_holdout_accuracy": _safe_float(test_row.get("accuracy")),
                    "final_holdout_ece": _safe_float(test_row.get("ece")),
                    "log_loss_improvement_vs_intercept": log_loss_improvement_vs_intercept,
                    "brier_improvement_vs_intercept": brier_improvement_vs_intercept,
                    "auc_improvement_vs_intercept": auc_improvement_vs_intercept,
                    "log_loss_gap_vs_recommended": log_loss_gap_vs_recommended,
                    "brier_gap_vs_recommended": brier_gap_vs_recommended,
                    "auc_gap_vs_recommended": auc_gap_vs_recommended,
                },
                stability_metrics={
                    "validation_rank": validation_rank,
                    "final_holdout_rank": test_rank,
                    "rank_shift_validation_to_final": (
                        int(test_rank - validation_rank) if validation_rank is not None and test_rank is not None else None
                    ),
                    "validation_to_test_log_loss_delta": (
                        _safe_float(test_row.get("log_loss")) - _safe_float(validation_row.get("log_loss"))
                        if validation_row is not None
                        and _safe_float(test_row.get("log_loss")) is not None
                        and _safe_float(validation_row.get("log_loss")) is not None
                        else None
                    ),
                    "validation_to_test_brier_delta": (
                        _safe_float(test_row.get("brier")) - _safe_float(validation_row.get("brier"))
                        if validation_row is not None
                        and _safe_float(test_row.get("brier")) is not None
                        and _safe_float(validation_row.get("brier")) is not None
                        else None
                    ),
                    "validation_to_test_auc_delta": (
                        _safe_float(test_row.get("auc")) - _safe_float(validation_row.get("auc"))
                        if validation_row is not None
                        and _safe_float(test_row.get("auc")) is not None
                        and _safe_float(validation_row.get("auc")) is not None
                        else None
                    ),
                    "bootstrap_delta_log_loss_mean_vs_recommended": (
                        _safe_float(bootstrap_row.get("delta_log_loss_mean")) if bootstrap_row is not None else None
                    ),
                    "bootstrap_delta_log_loss_prob_recommended_better": (
                        _safe_float(bootstrap_row.get("delta_log_loss_prob_reference_better"))
                        if bootstrap_row is not None
                        else None
                    ),
                    "bootstrap_delta_brier_mean_vs_recommended": (
                        _safe_float(bootstrap_row.get("delta_brier_mean")) if bootstrap_row is not None else None
                    ),
                    "bootstrap_delta_brier_prob_recommended_better": (
                        _safe_float(bootstrap_row.get("delta_brier_prob_reference_better"))
                        if bootstrap_row is not None
                        else None
                    ),
                    "active_parameter_ratio": active_parameter_ratio,
                },
                calibration_summary={
                    "final_holdout_ece": _safe_float(test_row.get("ece")),
                    "final_holdout_mce": _safe_float(test_row.get("mce")),
                    "calibration_alpha": _safe_float(test_row.get("calibration_alpha")),
                    "calibration_beta": _safe_float(test_row.get("calibration_beta")),
                    "brier_reliability": _safe_float(test_row.get("brier_reliability")),
                    "brier_resolution": _safe_float(test_row.get("brier_resolution")),
                    "brier_uncertainty": _safe_float(test_row.get("brier_uncertainty")),
                    "mean_abs_calibration_gap": _safe_float(test_row.get("mean_abs_calibration_gap")),
                    "max_abs_calibration_gap": _safe_float(test_row.get("max_abs_calibration_gap")),
                    "normalized_gini": _safe_float(test_row.get("normalized_gini")),
                    "top_decile_event_capture": _safe_float(test_row.get("top_decile_event_capture")),
                    "top_quantile_actual_lift": _safe_float(test_row.get("top_quantile_actual_lift")),
                    "top_vs_bottom_actual_lift_ratio": _safe_float(test_row.get("top_vs_bottom_actual_lift_ratio")),
                    "tossup_share_current_band": _safe_float(test_row.get("tossup_share_current_band")),
                },
                complement_summary={
                    "display_name": str(test_row.get("display_name") or metadata["display_name"]),
                    "family": metadata["family"],
                    "governance_note": metadata["governance_note"],
                    "params": str(test_row.get("params") or ""),
                    "fit_status": str(test_row.get("fit_status") or ""),
                    "fit_error": str(test_row.get("fit_error") or ""),
                    "recommended_for_next_stage": recommended_for_next_stage,
                    "next_stage_blocked_by_fixture_scope": bool(fixture_or_demo and model_name == recommended_model),
                    "benchmark_model": "intercept_only",
                    "benchmark_display_name": str(benchmark_row.get("display_name")) if benchmark_row is not None else "Intercept Only",
                    "recommended_model": recommended_model,
                    "recommendation_tier": recommendation_tier,
                    "recommendation_badges": badges,
                    "summary_note": (
                        "Fixture-slice screen leader only; rerun on the full immutable pregame MLB ledger before promotion review."
                        if recommendation_tier == "fixture_slice_screen_leader"
                        else (
                            "Best final-holdout named candidate in this research screen."
                            if recommendation_tier == "research_screen_leader"
                            else (
                                "Close challenger retained on the comparison watchlist."
                                if recommendation_tier == "challenge_watchlist"
                                else "Not recommended for the next stage in this comparison run."
                            )
                        )
                    ),
                },
                rejection_reasons=rejection_reasons,
            )
        )
    return scorecards


def _build_candidate_leaderboard(
    *,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    scorecards: list[CandidateScorecardRecord],
) -> pd.DataFrame:
    validation_lookup = {str(row["model_name"]): row for _, row in validation_metrics.iterrows()}
    validation_rank_lookup = _candidate_rank_lookup(validation_metrics)
    test_rank_lookup = _candidate_rank_lookup(test_metrics)
    scorecard_lookup = {record.model_name: record for record in scorecards}
    rows: list[dict[str, Any]] = []

    for _, test_row in _candidate_metric_rows(test_metrics).iterrows():
        model_name = str(test_row["model_name"])
        validation_row = validation_lookup.get(model_name)
        scorecard = scorecard_lookup.get(model_name)
        rejection_reasons = list(scorecard.rejection_reasons) if scorecard is not None else []
        recommendation_tier = (
            str(scorecard.complement_summary.get("recommendation_tier"))
            if scorecard is not None and isinstance(scorecard.complement_summary, dict)
            else "hold"
        )
        rows.append(
            {
                "model_name": model_name,
                "display_name": str(test_row.get("display_name") or model_name),
                "fit_status": str(test_row.get("fit_status") or ""),
                "validation_rank": validation_rank_lookup.get(model_name),
                "final_holdout_rank": test_rank_lookup.get(model_name),
                "rank_shift_validation_to_final": (
                    int(test_rank_lookup[model_name] - validation_rank_lookup[model_name])
                    if model_name in test_rank_lookup and model_name in validation_rank_lookup
                    else None
                ),
                "validation_log_loss": _safe_float(validation_row.get("log_loss")) if validation_row is not None else None,
                "validation_brier": _safe_float(validation_row.get("brier")) if validation_row is not None else None,
                "validation_auc": _safe_float(validation_row.get("auc")) if validation_row is not None else None,
                "final_holdout_log_loss": _safe_float(test_row.get("log_loss")),
                "final_holdout_brier": _safe_float(test_row.get("brier")),
                "final_holdout_auc": _safe_float(test_row.get("auc")),
                "final_holdout_ece": _safe_float(test_row.get("ece")),
                "final_holdout_mce": _safe_float(test_row.get("mce")),
                "recommendation_tier": recommendation_tier,
                "rejection_reasons": rejection_reasons,
            }
        )
    return pd.DataFrame(rows)


def _build_recommendation_surface(
    *,
    league: str,
    report_slug: str,
    comparison_decision: PromotionDecisionRecord,
    candidate_scorecards: list[CandidateScorecardRecord],
    candidate_leaderboard: pd.DataFrame,
) -> dict[str, Any]:
    decision = comparison_decision.to_dict()
    recommended_model = str(decision.get("recommended_model") or "")
    scorecard_lookup = {record.model_name: record for record in candidate_scorecards}
    recommended_scorecard = scorecard_lookup.get(recommended_model)
    top_rows = candidate_leaderboard.head(3).to_dict(orient="records") if not candidate_leaderboard.empty else []
    watchlist = [
        {
            "model_name": record.model_name,
            "display_name": str(record.complement_summary.get("display_name") or record.model_name),
            "tier": str(record.complement_summary.get("recommendation_tier") or "hold"),
            "final_holdout_log_loss": _safe_float(record.validation_metrics.get("final_holdout_log_loss")),
            "final_holdout_brier": _safe_float(record.validation_metrics.get("final_holdout_brier")),
            "rejection_reasons": list(record.rejection_reasons),
        }
        for record in candidate_scorecards
        if str(record.complement_summary.get("recommendation_tier") or "") == "challenge_watchlist"
    ][:3]

    if str(decision.get("status")) == "fixture_slice_screen_complete" and recommended_model != "intercept_only":
        next_actions = [
            "Rerun this candidate comparison on the full immutable pregame MLB ledger.",
            "Treat the selected model as a fixture-slice screen leader only.",
            "Do not enter promotion review until full-data validation evidence exists.",
        ]
    elif str(decision.get("status")) == "research_recommended" and recommended_model != "intercept_only":
        next_actions = [
            "Run research backtest with full pregame ledger checks.",
            "Keep at least one challenger in the next backtest sweep for robustness.",
            "Gate any promotion on written evidence, not only this comparison screen.",
        ]
    else:
        next_actions = [
            "Hold promotion and keep intercept benchmark in the lane.",
            "Expand feature diagnostics and challenger variants before the next run.",
            "Re-run candidate comparison after feature/pipeline adjustments.",
        ]

    return {
        "league": str(league).upper(),
        "report_slug": report_slug,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "decision": {
            "status": decision.get("status"),
            "recommended_model": recommended_model,
            "recommended_display_name": (
                str(recommended_scorecard.complement_summary.get("display_name"))
                if recommended_scorecard is not None
                else recommended_model
            ),
            "baseline_model": decision.get("baseline_model"),
            "rationale": decision.get("rationale"),
        },
        "top_candidates": top_rows,
        "challenger_watchlist": watchlist,
        "next_actions": next_actions,
    }


def _build_comparison_decision(
    *,
    candidate_scorecards: list[CandidateScorecardRecord],
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    candidate_scope_note: str,
    feature_pool_note: str,
    execution_metadata: dict[str, Any] | None = None,
) -> PromotionDecisionRecord:
    fixture_or_demo = _is_fixture_or_demo_execution(execution_metadata)
    best_test = _best_model_row(test_metrics)
    winner_model = str(best_test["model_name"])
    winner_display = str(best_test["display_name"])
    candidate_test = _candidate_metric_rows(test_metrics)
    best_named_candidate = candidate_test.iloc[0] if not candidate_test.empty else None
    intercept_row = test_metrics[test_metrics["model_name"] == "intercept_only"]
    intercept_best = intercept_row.iloc[0] if not intercept_row.empty else None
    rejected_models = {
        record.model_name: list(record.rejection_reasons)
        for record in candidate_scorecards
        if record.rejection_reasons
    }

    if winner_model == "intercept_only":
        recommended_model = "intercept_only"
        status = "research_hold"
        rationale = (
            "No named CAS candidate beat the intercept-only benchmark on the final holdout. "
            "Keep the comparison lane open rather than promoting a paper winner."
        )
    else:
        recommended_model = winner_model
        if fixture_or_demo:
            status = "fixture_slice_screen_complete"
            rationale = (
                f"{winner_display} led the named CAS candidates on the fixture-slice final holdout. "
                "Rerun on the full immutable pregame MLB ledger before research backtest or promotion review."
            )
        else:
            status = "research_recommended"
            rationale = (
                f"{winner_display} led the named CAS candidates on the final holdout. "
                "Advance it to research backtest and promotion review, but do not treat this comparison as a champion declaration."
            )

    top_rows = candidate_test.head(3)
    evidence = {
        "decision_scope": "candidate_model_comparison",
        "promotion_ready": False,
        "candidate_scope": candidate_scope_note,
        "feature_pool_note": feature_pool_note,
        "fixture_or_demo_execution": fixture_or_demo,
        "next_stage_blocked_by_fixture_scope": bool(fixture_or_demo and recommended_model != "intercept_only"),
        "final_holdout_winner": _row_metric_slice(best_test, ["model_name", "display_name", "log_loss", "brier", "auc", "ece"]),
        "best_named_candidate": _row_metric_slice(
            best_named_candidate,
            ["model_name", "display_name", "log_loss", "brier", "auc", "ece"],
        ),
        "intercept_only_benchmark": _row_metric_slice(
            intercept_best,
            ["model_name", "display_name", "log_loss", "brier", "auc", "ece"],
        ),
        "top_final_holdout_models": [
            _row_metric_slice(row, ["model_name", "display_name", "log_loss", "brier", "auc", "ece"])
            for _, row in top_rows.iterrows()
        ],
        "bootstrap_summary": [
            _row_metric_slice(
                row,
                [
                    "reference_model",
                    "comparison_model",
                    "delta_log_loss_mean",
                    "delta_log_loss_p025",
                    "delta_log_loss_p975",
                    "delta_log_loss_prob_reference_better",
                ],
            )
            for _, row in bootstrap_summary.head(3).iterrows()
        ],
        "validation_ranking": [
            _row_metric_slice(row, ["model_name", "display_name", "log_loss", "brier", "auc"])
            for _, row in _candidate_metric_rows(validation_metrics).head(3).iterrows()
        ],
    }
    if execution_metadata:
        evidence["execution_metadata"] = dict(execution_metadata)
    return PromotionDecisionRecord(
        recommended_model=recommended_model,
        baseline_model="intercept_only",
        status=status,
        rationale=rationale,
        evidence=evidence,
        rejected_models=rejected_models,
    )


def _write_report(
    *,
    cfg: AppConfig,
    report_path: Path,
    summary_path: Path,
    raw_feature_count: int,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_features: CandidateFeatureSets,
    final_features: CandidateFeatureSets,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    comparison_decision: PromotionDecisionRecord,
    candidate_scope_note: str,
    feature_pool_note: str,
    split_metadata: dict[str, Any],
    execution_metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    best_test = _best_model_row(test_metrics)
    candidate_validation = validation_metrics[validation_metrics["model_name"] != "intercept_only"].copy()
    candidate_test = test_metrics[test_metrics["model_name"] != "intercept_only"].copy()
    best_candidate_test = _best_model_row(candidate_test)
    try:
        best_candidate_validation = _best_model_row(candidate_validation)
    except RuntimeError:
        best_candidate_validation = best_candidate_test
    winner_model = str(best_test["model_name"])
    winner_display = str(best_test["display_name"])
    screening_summary = final_features.screening_frame["status"].value_counts().to_dict()
    fixture_or_demo = _is_fixture_or_demo_execution(execution_metadata)
    row_count_label = "Fixture-slice rows used" if fixture_or_demo else "Historical rows used"
    lines = [
        f"# {cfg.data.league} Candidate Model Comparison",
        "",
        "Protocol",
        "- Objective: maximize out-of-sample probability quality for home-win probabilities under proper scoring rules.",
        "- Guidance followed from the local CAS monograph sections on train/validation/test splitting (4.3), deviance and penalized fit comparisons (6.1-6.2), residual/nonlinearity/stability checks (6.3-6.4), holdout actual-vs-predicted/lift/ROC validation (7.1-7.3), and extension candidates (10.1-10.5).",
        (
            f"- Outer split `{split_metadata.get('split_label', 'custom')}`: "
            f"{float(split_metadata.get('train_fraction') or 0.0):.0%} train, "
            f"{float(split_metadata.get('validation_fraction') or 0.0):.0%} validation, "
            f"{float(split_metadata.get('test_fraction') or 0.0):.0%} final test, ordered by `start_time_utc`."
        ),
        "- Hyperparameters were tuned with rolling time-series CV inside the fit window for each phase.",
        f"- Candidate scope: {candidate_scope_note}",
        f"- Feature pool: {feature_pool_note}",
        "",
        "Data",
        f"- League: {cfg.data.league}",
        f"- {row_count_label}: {len(train_df) + len(validation_df) + len(test_df)}",
        f"- Data date range: {split_metadata.get('data_start') or 'n/a'} to {split_metadata.get('data_end') or 'n/a'}",
        f"- Final holdout date range: {split_metadata.get('test_start') or 'n/a'} to {split_metadata.get('test_end') or 'n/a'}",
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
        "- This comparison is a research screen only; it does not promote a production champion.",
        f"- Best final-holdout screen model: {winner_display} (`{winner_model}`)",
        f"- Best named candidate on the final holdout: {best_candidate_test['display_name']} (`{best_candidate_test['model_name']}`)",
        f"- Final test log loss of the best named candidate: {float(best_candidate_test['log_loss']):.6f}",
        f"- Final test Brier score of the best named candidate: {float(best_candidate_test['brier']):.6f}",
        f"- Final test AUC of the best named candidate: {float(best_candidate_test['auc']):.6f}",
        f"- Best validation candidate: {best_candidate_validation['display_name']} (`{best_candidate_validation['model_name']}`)",
    ]
    if execution_metadata:
        execution_scope = str(execution_metadata.get("execution_data_scope") or "unspecified")
        source_label = str(execution_metadata.get("source_label") or execution_scope)
        lines.extend(
            [
                "",
                "Execution Context",
                f"- Execution data scope: {execution_scope}",
                f"- Source label: {source_label}",
            ]
        )
        if execution_scope != "full_mlb_data":
            lines.append(
                "- Evidence label: bounded MLB slice/fixture evidence only; this is not a production-grade champion promotion."
            )
    if winner_model == "intercept_only":
        lines.extend(
            [
                "- Recommendation: hold the line. None of the named candidates beat the intercept-only benchmark on the final holdout under proper scoring rules.",
                f"- Among the named candidates, the least-bad test model was {best_candidate_test['display_name']} (`{best_candidate_test['model_name']}`), but it still underperformed the benchmark.",
            ]
        )
    else:
        lines.append(
            f"- Recommendation: advance {winner_display} (`{winner_model}`) into research backtest and promotion review if the goal is stronger out-of-sample probability quality among the tested options."
        )
    if not bootstrap_summary.empty:
        second_row = bootstrap_summary.iloc[0]
        lines.extend(
            [
                f"- Closest challenger on the test set: `{second_row['comparison_model']}`",
                f"- Mean log-loss delta versus winner (challenger minus winner): {float(second_row['delta_log_loss_mean']):.6f}",
                f"- 95% bootstrap interval for that log-loss delta: [{float(second_row['delta_log_loss_p025']):.6f}, {float(second_row['delta_log_loss_p975']):.6f}]",
            ]
        )
    lines.extend(
        [
            f"- Contract summary: `{summary_path.name}`",
            f"- Recorded recommendation status: `{comparison_decision.status}`",
            f"- Recommendation rationale: {comparison_decision.rationale}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    return winner_model, winner_display


def run_candidate_model_comparison(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    candidate_models: list[str] | None = None,
    feature_pool: str = FEATURE_POOL_FULL_SCREENED,
    feature_map_model: str = "glm_ridge",
    structured_glm_spec_path: str | None = None,
    structured_glm_slate: str | None = None,
    structured_glm_width_variant: str | None = None,
    train_fraction: float = 0.4,
    validation_fraction: float = 0.3,
    split_label: str = "outer_40_30_30",
    execution_metadata: dict[str, Any] | None = None,
) -> ComparisonRunResult:
    feature_pool_token = str(feature_pool or FEATURE_POOL_FULL_SCREENED).strip().lower()
    if feature_pool_token == FEATURE_POOL_RESEARCH_BROAD:
        research_paths = resolve_research_paths(cfg)
        features_df = load_features_dataframe(str(research_paths.processed_dir))
    else:
        features_df = load_features_dataframe(cfg.paths.processed_dir)
    all_raw_features = select_feature_columns(features_df)
    leakage_issues = run_leakage_checks(features_df, feature_columns=all_raw_features)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed before model comparison: {leakage_issues}")

    candidate_model_set: set[str] | None = None
    if candidate_models:
        normalized = {str(model).strip() for model in candidate_models if str(model).strip()}
        bad = sorted(normalized - CANDIDATE_MODEL_NAMES)
        if bad:
            raise ValueError(f"Unknown candidate model names: {bad}. Valid={sorted(CANDIDATE_MODEL_NAMES)}")
        if not normalized:
            raise ValueError("candidate_models must include at least one valid model name")
        candidate_model_set = normalized

    if feature_pool_token not in {FEATURE_POOL_FULL_SCREENED, FEATURE_POOL_PRODUCTION_MODEL_MAP, FEATURE_POOL_RESEARCH_BROAD}:
        raise ValueError(
            f"Unknown feature_pool='{feature_pool}'. "
            f"Valid={[FEATURE_POOL_FULL_SCREENED, FEATURE_POOL_PRODUCTION_MODEL_MAP, FEATURE_POOL_RESEARCH_BROAD]}"
        )

    if feature_pool_token == FEATURE_POOL_PRODUCTION_MODEL_MAP:
        model_feature_map = load_model_feature_map(cfg.data.league)
        requested_features = [str(col) for col in model_feature_map.get(str(feature_map_model), []) if str(col).strip()]
        if not requested_features:
            raise ValueError(
                f"Production model feature map for league={cfg.data.league} does not contain '{feature_map_model}'"
            )
        missing = [feature for feature in requested_features if feature not in all_raw_features]
        if missing:
            raise ValueError(
                f"Production feature map '{feature_map_model}' includes columns not available after leakage bans: {missing}"
            )
        raw_features = requested_features
        feature_pool_note = (
            f"Restricted to the production model feature map for `{feature_map_model}` "
            f"after leakage bans ({len(raw_features)} raw features)."
        )
    elif feature_pool_token == FEATURE_POOL_RESEARCH_BROAD:
        raw_features = all_raw_features
        feature_pool_note = (
            "Built from the research dataset's broad numeric feature pool after leakage bans, "
            "without applying production model-map pruning."
        )
    else:
        raw_features = all_raw_features
        feature_pool_note = (
            "Built from the full numeric feature pool after leakage bans, not from the repo's "
            "production `glm_feature_subset` or per-model feature map."
        )

    structured_glm_resolution = resolve_structured_glm_experiment(
        league=cfg.data.league,
        available_features=raw_features,
        spec_path=structured_glm_spec_path,
        slate_name=structured_glm_slate,
        width_variant=structured_glm_width_variant,
    )
    feature_pool_note = structured_glm_resolution.extend_feature_pool_note(
        feature_pool_note,
        connector=" Plus ",
        suffix=".",
    )

    if candidate_model_set:
        candidate_scope_note = ", ".join(sorted(candidate_model_set))
    else:
        candidate_scope_note = "all registered candidate families"

    historical_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    if train_fraction <= 0.0 or validation_fraction < 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "Candidate comparison split fractions must satisfy "
            "0 < train_fraction, 0 <= validation_fraction, and train+validation < 1."
        )
    train_df, validation_df, test_df = _time_ordered_split(
        historical_df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    data_start, data_end = _date_bounds(historical_df)
    train_start, train_end = _date_bounds(train_df)
    validation_start, validation_end = _date_bounds(validation_df)
    test_start, test_end = _date_bounds(test_df)
    split_metadata = {
        "split_label": str(split_label or "custom"),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1.0 - train_fraction - validation_fraction),
        "n_historical": int(len(historical_df)),
        "n_train": int(len(train_df)),
        "n_validation": int(len(validation_df)),
        "n_test": int(len(test_df)),
        "data_start": data_start,
        "data_end": data_end,
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
        "split_method": "time_ordered_by_start_time_utc",
    }
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
            cv_splits=max(2, int(cfg.research.inner_folds if feature_pool_token == FEATURE_POOL_RESEARCH_BROAD else cfg.modeling.cv_splits)),
            candidate_models=candidate_model_set,
            structured_glm_resolution=structured_glm_resolution,
        )
        fit_plus_validation = pd.concat([train_df, validation_df], ignore_index=True).sort_values("start_time_utc").reset_index(drop=True)
        final_features, test_metrics, test_predictions, test_fit_stats, final_cv = _phase_evaluation(
            split_name="test",
            fit_df=fit_plus_validation,
            eval_df=test_df,
            raw_features=raw_features,
            cv_splits=max(2, int(cfg.research.inner_folds if feature_pool_token == FEATURE_POOL_RESEARCH_BROAD else cfg.modeling.cv_splits)),
            candidate_models=candidate_model_set,
            structured_glm_resolution=structured_glm_resolution,
        )

    bootstrap_summary = _bootstrap_against_best(
        test_predictions,
        test_metrics,
        n_bootstrap=max(100, int(bootstrap_samples)),
        random_seed=int(cfg.modeling.random_seed),
    )

    stamp = datetime.now().strftime("%Y-%m-%d")
    slug = report_slug or f"{stamp}_candidate_model_comparison"
    primary_dir = _resolve_comparison_report_dirs(cfg, league=str(cfg.data.league))
    prefix = _file_prefix(cfg.data.league, slug)

    validation_metrics_path = primary_dir / f"{prefix}_validation_metrics.csv"
    test_metrics_path = primary_dir / f"{prefix}_test_metrics.csv"
    bootstrap_path = primary_dir / f"{prefix}_bootstrap.csv"
    screening_path = primary_dir / f"{prefix}_feature_screening.csv"
    nonlinearity_path = primary_dir / f"{prefix}_nonlinearity.csv"
    validation_predictions_path = primary_dir / f"{prefix}_validation_predictions.csv"
    test_predictions_path = primary_dir / f"{prefix}_test_predictions.csv"
    fit_stats_path = primary_dir / f"{prefix}_fit_stats.csv"
    cv_path = primary_dir / f"{prefix}_cv_summary.csv"
    candidate_scorecards_path = primary_dir / f"{prefix}_candidate_scorecards.csv"
    candidate_scorecards_contract_path = primary_dir / f"{prefix}_candidate_scorecards.json"
    leaderboard_path = primary_dir / f"{prefix}_candidate_leaderboard.csv"
    leaderboard_json_path = primary_dir / f"{prefix}_candidate_leaderboard.json"
    recommendation_path = primary_dir / f"{prefix}_recommendation.json"
    recommendation_surface_path = primary_dir / f"{prefix}_recommendation_surface.json"
    artifact_manifest_path = primary_dir / f"{prefix}_artifact_manifest.json"
    summary_path = primary_dir / f"{prefix}_comparison_summary.json"
    report_path = primary_dir / f"{prefix}_summary.md"

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

    candidate_scorecards = _build_candidate_scorecards(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_fit_stats=test_fit_stats,
        bootstrap_summary=bootstrap_summary,
        recommended_model=str(_best_model_row(test_metrics)["model_name"]),
        execution_metadata=execution_metadata,
    )
    comparison_decision = _build_comparison_decision(
        candidate_scorecards=candidate_scorecards,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
        candidate_scope_note=candidate_scope_note,
        feature_pool_note=feature_pool_note,
        execution_metadata=execution_metadata,
    )
    candidate_scorecard_payload = [record.to_dict() for record in candidate_scorecards]
    pd.json_normalize(candidate_scorecard_payload, sep="__").to_csv(candidate_scorecards_path, index=False)
    candidate_scorecards_contract_path.write_text(to_json(candidate_scorecard_payload) + "\n")
    recommendation_path.write_text(to_json(comparison_decision.to_dict()) + "\n")
    leaderboard = _build_candidate_leaderboard(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        scorecards=candidate_scorecards,
    )
    leaderboard.to_csv(leaderboard_path, index=False)
    leaderboard_records = leaderboard.to_dict(orient="records")
    leaderboard_json_path.write_text(to_json(leaderboard_records) + "\n")
    recommendation_surface = _build_recommendation_surface(
        league=str(cfg.data.league),
        report_slug=slug,
        comparison_decision=comparison_decision,
        candidate_scorecards=candidate_scorecards,
        candidate_leaderboard=leaderboard,
    )
    recommendation_surface_path.write_text(to_json(recommendation_surface) + "\n")
    artifact_manifest_payload = {
        "report_slug": slug,
        "league": str(cfg.data.league).upper(),
        "artifact_root": str(primary_dir),
        "artifacts": {
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "validation_metrics_path": str(validation_metrics_path),
            "test_metrics_path": str(test_metrics_path),
            "bootstrap_path": str(bootstrap_path),
            "fit_stats_path": str(fit_stats_path),
            "cv_path": str(cv_path),
            "screening_path": str(screening_path),
            "nonlinearity_path": str(nonlinearity_path),
            "validation_predictions_path": str(validation_predictions_path),
            "test_predictions_path": str(test_predictions_path),
            "candidate_scorecards_path": str(candidate_scorecards_path),
            "candidate_scorecards_contract_path": str(candidate_scorecards_contract_path),
            "leaderboard_path": str(leaderboard_path),
            "leaderboard_json_path": str(leaderboard_json_path),
            "recommendation_path": str(recommendation_path),
            "recommendation_surface_path": str(recommendation_surface_path),
        },
    }
    artifact_manifest_path.write_text(to_json(artifact_manifest_payload) + "\n")
    summary_payload = CandidateComparisonContract(
        report_slug=slug,
        league=str(cfg.data.league).upper(),
        target_name=COMPARISON_TARGET_NAME,
        distribution=COMPARISON_DISTRIBUTION,
        link_function=COMPARISON_LINK_FUNCTION,
        candidate_models=[record.model_name for record in candidate_scorecards],
        candidate_scorecards=candidate_scorecards,
        promotion_decision=comparison_decision,
        artifacts={
            "report_path": str(report_path),
            "validation_metrics_path": str(validation_metrics_path),
            "test_metrics_path": str(test_metrics_path),
            "bootstrap_path": str(bootstrap_path),
            "fit_stats_path": str(fit_stats_path),
            "cv_path": str(cv_path),
            "screening_path": str(screening_path),
            "nonlinearity_path": str(nonlinearity_path),
            "validation_predictions_path": str(validation_predictions_path),
            "test_predictions_path": str(test_predictions_path),
            "candidate_scorecards_path": str(candidate_scorecards_path),
            "candidate_scorecards_contract_path": str(candidate_scorecards_contract_path),
            "leaderboard_path": str(leaderboard_path),
            "leaderboard_json_path": str(leaderboard_json_path),
            "recommendation_path": str(recommendation_path),
            "recommendation_surface_path": str(recommendation_surface_path),
            "artifact_manifest_path": str(artifact_manifest_path),
        },
        metadata={
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "candidate_scope_note": candidate_scope_note,
            "feature_pool_note": feature_pool_note,
            "artifact_root": str(primary_dir),
            "raw_feature_count": int(len(raw_features)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "split": split_metadata,
            "screening_summary": final_features.screening_frame["status"].value_counts().to_dict(),
            "final_nonlinearity_summary": dict(final_features.nonlinearity_summary),
            "recommended_display_name": str(_best_model_row(test_metrics)["display_name"]),
            "leaderboard_preview": leaderboard_records[:5],
            "recommendation_surface": recommendation_surface,
            "execution_metadata": dict(execution_metadata or {}),
        },
    ).to_dict()
    summary_path.write_text(to_json(summary_payload) + "\n")

    recommendation_model, recommendation_display_name = _write_report(
        cfg=cfg,
        report_path=report_path,
        summary_path=summary_path,
        raw_feature_count=len(raw_features),
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        validation_features=validation_features,
        final_features=final_features,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
        comparison_decision=comparison_decision,
        candidate_scope_note=candidate_scope_note,
        feature_pool_note=feature_pool_note,
        split_metadata=split_metadata,
        execution_metadata=execution_metadata,
    )

    return ComparisonRunResult(
        league=cfg.data.league,
        report_slug=slug,
        report_path=report_path,
        summary_path=summary_path,
        candidate_scorecards_path=candidate_scorecards_path,
        candidate_scorecards_contract_path=candidate_scorecards_contract_path,
        recommendation_path=recommendation_path,
        validation_metrics_path=validation_metrics_path,
        test_metrics_path=test_metrics_path,
        bootstrap_path=bootstrap_path,
        fit_stats_path=fit_stats_path,
        cv_path=cv_path,
        recommendation_model=recommendation_model,
        recommendation_display_name=recommendation_display_name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
        scorecards_path=candidate_scorecards_contract_path,
        leaderboard_path=leaderboard_path,
        leaderboard_json_path=leaderboard_json_path,
        recommendation_surface_path=recommendation_surface_path,
        artifact_manifest_path=artifact_manifest_path,
    )
