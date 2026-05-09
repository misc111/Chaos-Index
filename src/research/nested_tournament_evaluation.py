"""Candidate tuning and validation-lane evaluation for nested tournaments.

The functions here are the statistical workbench for one candidate or one
intra-family lane.  They produce rows and prediction series only; tournament
runners decide how those rows are aggregated and written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.utils import ensure_dir
from src.evaluation.calibration import calibration_alpha_beta
from src.evaluation.metrics import metric_bundle
from src.evaluation.validation_classification import validate_logistic_probability_model
from src.governance.evidence import model_evidence_fields
from src.research.nested_tournament_candidates import _candidate_specs, _params_text
from src.research.nested_tournament_contracts import NestedCandidateSpec, _safe_float, _safe_json, _time_ordered_split
from src.research.nested_tournament_features import _screened_feature_sets

def _time_series_splits(df: pd.DataFrame, *, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    if n < 24:
        return []
    split_count = max(2, int(n_splits))
    min_train = min(max(20, n // 2), max(n - 10, 10))
    validation_size = max(5, (n - min_train) // split_count)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(split_count):
        train_end = min_train + fold * validation_size
        valid_end = train_end + validation_size
        if fold == split_count - 1:
            valid_end = n
        if train_end >= n or valid_end <= train_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, valid_end)))
    return splits


def _tune_candidate(
    spec: NestedCandidateSpec,
    *,
    fit_df: pd.DataFrame,
    cv_splits: int,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    splits = _time_series_splits(fit_df.sort_values("start_time_utc").reset_index(drop=True), n_splits=cv_splits)
    if not splits:
        return None, pd.DataFrame()
    ordered = fit_df.sort_values("start_time_utc").reset_index(drop=True)
    for params in spec.param_grid:
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = ordered.iloc[tr_idx].copy()
            va = ordered.iloc[va_idx].copy()
            if va[spec.target.target_col].nunique(dropna=True) < 2:
                continue
            try:
                model = spec.builder(dict(params))
                model.fit(tr, target_col=spec.target.target_col)
                p = model.predict_proba(va)
                metrics = metric_bundle(va[spec.target.target_col].astype(int).to_numpy(), p)
                rows.append(
                    {
                        "target_name": spec.target.target_name,
                        "candidate_key": spec.candidate_key,
                        "model_name": spec.model_name,
                        "variant_key": spec.variant_key,
                        "params": _params_text(dict(params)),
                        "fold": fold,
                        "status": "ok",
                        "mean_log_loss": float(metrics["log_loss"]),
                        "mean_brier": float(metrics["brier"]),
                        "mean_auc": float(metrics["auc"]),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "target_name": spec.target.target_name,
                        "candidate_key": spec.candidate_key,
                        "model_name": spec.model_name,
                        "variant_key": spec.variant_key,
                        "params": _params_text(dict(params)),
                        "fold": fold,
                        "status": "failed",
                        "mean_log_loss": np.nan,
                        "mean_brier": np.nan,
                        "mean_auc": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    folds = pd.DataFrame(rows)
    ok = folds[folds["status"] == "ok"].copy() if not folds.empty else pd.DataFrame()
    if ok.empty:
        return None, folds
    summary = (
        ok.groupby("params", as_index=False)
        .agg(mean_log_loss=("mean_log_loss", "mean"), mean_brier=("mean_brier", "mean"), mean_auc=("mean_auc", "mean"))
        .sort_values(["mean_log_loss", "mean_brier", "mean_auc", "params"], ascending=[True, True, False, True])
    )
    params_text = str(summary.iloc[0]["params"])
    params = next((dict(params) for params in spec.param_grid if _params_text(dict(params)) == params_text), None)
    return params, folds


def _diagnostic_artifacts(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    out_dir: Path,
    prefix: str,
    model: Any | None = None,
    eval_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    validation = validate_logistic_probability_model(
        y_true,
        p_pred,
        bins=10,
        plot_dir=out_dir / "plots",
        plot_prefix=prefix,
    )
    paths: dict[str, str] = {}
    for key in (
        "quantile_curve",
        "actual_vs_predicted_curve",
        "lift_curve",
        "lorenz_curve",
        "roc_curve",
        "operating_points",
        "tossup_sweep",
    ):
        value = validation.get(key)
        if isinstance(value, pd.DataFrame):
            path = out_dir / f"{prefix}_{key}.csv"
            value.to_csv(path, index=False)
            paths[key] = str(path)
    summaries = {
        "quantile_summary": validation.get("quantile_summary"),
        "actual_vs_predicted_summary": validation.get("actual_vs_predicted_summary"),
        "lift_summary": validation.get("lift_summary"),
        "lorenz_summary": validation.get("lorenz_summary"),
        "roc_summary": validation.get("roc_summary"),
        "tossup_summary": validation.get("tossup_summary"),
        "applicability_summary": validation.get("applicability_summary"),
    }
    if model is not None and eval_df is not None and hasattr(model, "margin_diagnostics"):
        margin_summary = model.margin_diagnostics(eval_df)
        summaries["margin_summary"] = margin_summary
        margin_path = out_dir / f"{prefix}_margin_summary.json"
        _safe_json(margin_path, margin_summary)
        paths["margin_summary"] = str(margin_path)
    summary_path = out_dir / f"{prefix}_summary.json"
    _safe_json(summary_path, summaries)
    paths["summary"] = str(summary_path)
    for plot_key, plot_path in dict(validation.get("plot_paths") or {}).items():
        paths[plot_key] = str(plot_path)
    return summaries, paths


def _metric_row(
    *,
    spec: NestedCandidateSpec,
    split: str,
    params: dict[str, Any],
    y_true: np.ndarray,
    p_pred: np.ndarray,
    fit_stats: Any,
    cv_folds: pd.DataFrame,
    diagnostic_summary: dict[str, Any],
    diagnostic_paths: dict[str, str],
) -> dict[str, Any]:
    metrics = metric_bundle(y_true, p_pred)
    calibration = calibration_alpha_beta(y_true, p_pred)
    quantile = dict(diagnostic_summary.get("quantile_summary") or {})
    lift = dict(diagnostic_summary.get("lift_summary") or {})
    lorenz = dict(diagnostic_summary.get("lorenz_summary") or {})
    roc = dict(diagnostic_summary.get("roc_summary") or {})
    applicability = dict(diagnostic_summary.get("applicability_summary") or {})
    margin = dict(diagnostic_summary.get("margin_summary") or {})
    cv_ok = cv_folds[cv_folds["status"] == "ok"].copy() if not cv_folds.empty else pd.DataFrame()
    cv_mean_log_loss = _safe_float(cv_ok["mean_log_loss"].mean()) if not cv_ok.empty else None
    validation_to_cv_delta = (float(metrics["log_loss"]) - cv_mean_log_loss) if cv_mean_log_loss is not None else None
    evidence_fields = model_evidence_fields(
        spec.model_name,
        target_name=spec.target.target_name,
        target_col=spec.target.target_col,
        market=spec.target.market,
    )
    return {
        "target_name": spec.target.target_name,
        "target_col": spec.target.target_col,
        "market": spec.target.market,
        "split": split,
        "candidate_key": spec.candidate_key,
        "model_name": spec.model_name,
        "display_name": spec.display_name,
        "variant_key": spec.variant_key,
        "variant_display_name": spec.variant_display_name,
        "feature_source": spec.feature_source,
        "params": _params_text(params),
        "n_features": len(spec.features),
        "active_parameter_count": int(fit_stats.active_parameter_count) if fit_stats is not None else None,
        "log_loss": float(metrics["log_loss"]),
        "brier": float(metrics["brier"]),
        "accuracy": float(metrics["accuracy"]),
        "auc": _safe_float(metrics["auc"]),
        "ece": _safe_float(quantile.get("mean_abs_calibration_gap")),
        "calibration_alpha": _safe_float(calibration.get("calibration_alpha")),
        "calibration_beta": _safe_float(calibration.get("calibration_beta")),
        "max_abs_calibration_gap": _safe_float(quantile.get("max_abs_calibration_gap")),
        "top_vs_bottom_actual_lift_ratio": _safe_float(lift.get("top_vs_bottom_actual_lift_ratio")),
        "top_quantile_actual_lift": _safe_float(lift.get("top_quantile_actual_lift")),
        "normalized_gini": _safe_float(lorenz.get("normalized_gini")),
        "top_decile_event_capture": _safe_float(lorenz.get("top_decile_event_capture")),
        "roc_auc": _safe_float(roc.get("auroc")),
        "has_class_contrast": bool(applicability.get("has_class_contrast")),
        "cv_mean_log_loss": cv_mean_log_loss,
        "validation_to_cv_log_loss_delta": validation_to_cv_delta,
        "diagnostic_artifacts": diagnostic_paths,
        "margin_diagnostics": margin,
        "fit_status": "ok",
        "fit_error": "",
        **evidence_fields,
    }


def _failed_row(spec: NestedCandidateSpec, *, split: str, error: str) -> dict[str, Any]:
    evidence_fields = model_evidence_fields(
        spec.model_name,
        target_name=spec.target.target_name,
        target_col=spec.target.target_col,
        market=spec.target.market,
    )
    return {
        "target_name": spec.target.target_name,
        "target_col": spec.target.target_col,
        "market": spec.target.market,
        "split": split,
        "candidate_key": spec.candidate_key,
        "model_name": spec.model_name,
        "display_name": spec.display_name,
        "variant_key": spec.variant_key,
        "variant_display_name": spec.variant_display_name,
        "feature_source": spec.feature_source,
        "params": "",
        "n_features": len(spec.features),
        "active_parameter_count": None,
        "log_loss": np.nan,
        "brier": np.nan,
        "accuracy": np.nan,
        "auc": np.nan,
        "ece": np.nan,
        "calibration_alpha": np.nan,
        "calibration_beta": np.nan,
        "max_abs_calibration_gap": np.nan,
        "top_vs_bottom_actual_lift_ratio": np.nan,
        "top_quantile_actual_lift": np.nan,
        "normalized_gini": np.nan,
        "top_decile_event_capture": np.nan,
        "roc_auc": np.nan,
        "has_class_contrast": False,
        "cv_mean_log_loss": None,
        "validation_to_cv_log_loss_delta": None,
        "diagnostic_artifacts": {},
        "fit_status": "failed",
        "fit_error": error,
        **evidence_fields,
    }


def _evaluate_candidate(
    spec: NestedCandidateSpec,
    *,
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    params: dict[str, Any],
    split: str,
    cv_folds: pd.DataFrame,
    diagnostics_root: Path,
) -> tuple[dict[str, Any], pd.Series | None]:
    try:
        model = spec.builder(params)
        model.fit(fit_df, target_col=spec.target.target_col)
        p_pred = model.predict_proba(eval_df)
        y_true = eval_df[spec.target.target_col].astype(int).to_numpy()
        prefix = spec.candidate_key.replace("::", "__")
        diagnostic_summary, diagnostic_paths = _diagnostic_artifacts(
            y_true=y_true,
            p_pred=p_pred,
            out_dir=ensure_dir(diagnostics_root / spec.target.target_name / split / spec.model_name / spec.variant_key),
            prefix=prefix,
            model=model,
            eval_df=eval_df,
        )
        fit_stats = model.fit_statistics()
        row = _metric_row(
            spec=spec,
            split=split,
            params=params,
            y_true=y_true,
            p_pred=p_pred,
            fit_stats=fit_stats,
            cv_folds=cv_folds,
            diagnostic_summary=diagnostic_summary,
            diagnostic_paths=diagnostic_paths,
        )
        prediction = pd.Series(p_pred, name=spec.candidate_key)
        return row, prediction
    except Exception as exc:
        return _failed_row(spec, split=split, error=f"{type(exc).__name__}: {exc}"), None

def _evaluate_validation_lane(
    *,
    target_result: Any,
    model_name: str,
    raw_features: list[str],
    cv_splits: int,
    train_fraction: float,
    validation_fraction: float,
    structured_glm_spec_path: str | Path | None,
    diagnostics_root: Path,
    lane_root: Path,
) -> dict[str, Any]:
    target = target_result.definition
    target_df = target_result.frame
    train_df, validation_df, test_df = _time_ordered_split(
        target_df,
        target_col=target.target_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    variant_rows: list[dict[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []
    if train_df.empty or validation_df.empty or test_df.empty:
        path = ensure_dir(lane_root) / "validation_leaderboard.csv"
        pd.DataFrame().to_csv(path, index=False)
        return {
            "target_name": target.target_name,
            "model_name": model_name,
            "validation_leaderboard_path": str(path),
            "variant_rows": variant_rows,
            "cv_folds": cv_frames,
            "status": "coverage-blocked",
        }
    feature_sets = _screened_feature_sets(train_df, target_col=target.target_col, raw_features=raw_features)
    specs = _candidate_specs(
        target=target,
        feature_sets=feature_sets,
        candidate_models={model_name},
        structured_glm_spec_path=structured_glm_spec_path,
    )
    for spec in specs:
        best_params, cv_folds = _tune_candidate(spec, fit_df=train_df, cv_splits=cv_splits)
        if not cv_folds.empty:
            cv_frames.append(cv_folds)
        if best_params is None:
            variant_rows.append(_failed_row(spec, split="validation", error="no_successful_cv_folds"))
            continue
        row, _ = _evaluate_candidate(
            spec,
            fit_df=train_df,
            eval_df=validation_df,
            params=best_params,
            split="validation",
            cv_folds=cv_folds,
            diagnostics_root=diagnostics_root,
        )
        variant_rows.append(row)
    path = ensure_dir(lane_root) / "validation_leaderboard.csv"
    pd.DataFrame(variant_rows).to_csv(path, index=False)
    return {
        "target_name": target.target_name,
        "model_name": model_name,
        "validation_leaderboard_path": str(path),
        "variant_rows": variant_rows,
        "cv_folds": cv_frames,
        "status": "ok",
    }
