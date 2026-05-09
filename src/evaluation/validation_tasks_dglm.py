"""DGLM Margin validation diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.evaluation.validation_artifacts import ValidationOutputs, validation_path as _validation_path
from src.evaluation.validation_context import ValidationContext, holdout_df as _holdout_df
from src.evaluation.validation_task_support import (
    ValidationTaskResult,
    _binary_holdout_profile,
    _primary_model_key,
    _probability_model_family_applicability,
    _task_result,
    _unsupported_task_result,
)


def _is_dglm_margin_primary(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) == "dglm_margin"


def _dglm_dispersion_bins(residuals: pd.DataFrame, *, bins: int = 5) -> pd.DataFrame:
    required = {"predicted_sd", "predicted_variance", "squared_residual", "abs_residual", "standardized_residual"}
    if residuals.empty or not required.issubset(residuals.columns):
        return pd.DataFrame()
    work = residuals.copy()
    work["dispersion_bin"] = pd.qcut(
        pd.to_numeric(work["predicted_sd"], errors="coerce"),
        q=min(int(bins), int(work["predicted_sd"].nunique(dropna=True))),
        labels=False,
        duplicates="drop",
    )
    work = work[work["dispersion_bin"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bin_id, bucket in work.groupby("dispersion_bin", sort=True):
        residual_mse = float(bucket["squared_residual"].mean())
        predicted_variance = float(bucket["predicted_variance"].mean())
        within_1sd = float((bucket["standardized_residual"].abs() <= 1.0).mean())
        within_2sd = float((bucket["standardized_residual"].abs() <= 2.0).mean())
        rows.append(
            {
                "dispersion_bin": int(bin_id) + 1,
                "n": int(len(bucket)),
                "predicted_sd_min": float(bucket["predicted_sd"].min()),
                "predicted_sd_max": float(bucket["predicted_sd"].max()),
                "mean_predicted_sd": float(bucket["predicted_sd"].mean()),
                "mean_predicted_variance": predicted_variance,
                "residual_mse": residual_mse,
                "variance_to_residual_mse_ratio": float(residual_mse / max(predicted_variance, 1e-6)),
                "mean_abs_residual": float(bucket["abs_residual"].mean()),
                "standardized_residual_mean": float(bucket["standardized_residual"].mean()),
                "standardized_residual_sd": float(bucket["standardized_residual"].std(ddof=0)),
                "within_1sd_share": within_1sd,
                "within_2sd_share": within_2sd,
            }
        )
    return pd.DataFrame(rows)


def _task_dglm_margin_diagnostics(ctx: ValidationContext) -> ValidationTaskResult:
    if ctx.glm is None or not hasattr(ctx.glm, "margin_diagnostics"):
        return _unsupported_task_result(
            ctx,
            task_name="dglm_margin_diagnostics",
            note="DGLM margin diagnostics require a fitted dglm_margin primary model.",
        )

    holdout = _holdout_df(ctx)
    y = holdout["home_win"].astype(int).to_numpy()
    p = ctx.glm.predict_proba(holdout)
    probability_metrics = metric_bundle(y, p)
    margin_summary: dict[str, Any] = ctx.glm.margin_diagnostics(holdout)
    residuals = (
        ctx.glm.margin_prediction_frame(holdout)
        if hasattr(ctx.glm, "margin_prediction_frame")
        else pd.DataFrame()
    )
    dispersion_bins = _dglm_dispersion_bins(residuals)
    holdout_profile = _binary_holdout_profile(y)
    payload = {
        "status": margin_summary.get("status", "unknown"),
        "margin_summary": margin_summary,
        "residual_diagnostics": margin_summary.get("residual_diagnostics", {}),
        "dispersion_diagnostics": margin_summary.get("dispersion_diagnostics", {}),
        "probability_bridge_metrics": {
            "log_loss": probability_metrics["log_loss"],
            "brier": probability_metrics["brier"],
            "accuracy": probability_metrics["accuracy"],
            "auc": probability_metrics["auc"],
        },
        "holdout_profile": holdout_profile,
        "promotion_gate": "separate_margin_and_bridge_validation_required",
        "model_family_applicability": _probability_model_family_applicability(
            ctx,
            task_name="dglm_margin_diagnostics",
            requires_class_contrast=True,
            class_contrast_available=bool(holdout_profile["has_class_contrast"]),
            n_obs=int(holdout_profile["n_obs"]),
        ),
        "interpretation": (
            "This task validates DGLM Margin as a margin-plus-dispersion model before the derived "
            "moneyline probability is considered for promotion evidence."
        ),
    }
    row = pd.DataFrame(
        [
            {
                "status": payload["status"],
                "bridge": margin_summary.get("bridge"),
                "bridge_status": margin_summary.get("bridge_status"),
                "margin_bias": margin_summary.get("margin_bias"),
                "margin_mae": margin_summary.get("margin_mae"),
                "margin_rmse": margin_summary.get("margin_rmse"),
                "residual_sd": margin_summary.get("residual_sd"),
                "mean_predicted_sd": margin_summary.get("mean_predicted_sd"),
                "variance_to_residual_mse_ratio": margin_summary.get("variance_to_residual_mse_ratio"),
                "standardized_residual_mean": margin_summary.get("standardized_residual_mean"),
                "standardized_residual_sd": margin_summary.get("standardized_residual_sd"),
                "mean_abs_standardized_residual": (
                    (margin_summary.get("residual_diagnostics") or {}).get("mean_abs_standardized_residual")
                ),
                "within_1sd_share": margin_summary.get("within_1sd_share"),
                "within_2sd_share": margin_summary.get("within_2sd_share"),
                "within_1sd_gap_vs_normal": (
                    (margin_summary.get("dispersion_diagnostics") or {}).get("within_1sd_gap_vs_normal")
                ),
                "within_2sd_gap_vs_normal": (
                    (margin_summary.get("dispersion_diagnostics") or {}).get("within_2sd_gap_vs_normal")
                ),
                "abs_residual_predicted_sd_corr": margin_summary.get("abs_residual_predicted_sd_corr"),
                "bridge_log_loss": probability_metrics["log_loss"],
                "bridge_brier": probability_metrics["brier"],
                "bridge_auc": probability_metrics["auc"],
            }
        ]
    )

    out = ValidationOutputs()
    out.add_json(
        section="dglm_margin_diagnostics_summary",
        file_name=_validation_path("diagnostics", "dglm_margin", "validation_dglm_margin_summary.json"),
        payload=payload,
    )
    out.add_csv(
        section="dglm_margin_diagnostics_metrics",
        file_name=_validation_path("diagnostics", "dglm_margin", "validation_dglm_margin_metrics.csv"),
        rows=row,
    )
    out.add_csv(
        section="dglm_margin_residuals",
        file_name=_validation_path("diagnostics", "dglm_margin", "validation_dglm_margin_residuals.csv"),
        rows=residuals,
    )
    out.add_json(
        section="dglm_margin_residual_diagnostics",
        file_name=_validation_path("diagnostics", "dglm_margin", "validation_dglm_margin_residual_diagnostics.json"),
        payload={
            "status": margin_summary.get("status", "unknown"),
            "model_name": "dglm_margin",
            "residual_diagnostics": margin_summary.get("residual_diagnostics", {}),
        },
    )
    out.add_csv(
        section="dglm_margin_dispersion_bins",
        file_name=_validation_path("diagnostics", "dglm_margin", "validation_dglm_margin_dispersion_bins.csv"),
        rows=dispersion_bins,
    )
    out.add_json(
        section="dglm_margin_dispersion_diagnostics",
        file_name=_validation_path("diagnostics", "dglm_margin", "validation_dglm_margin_dispersion_diagnostics.json"),
        payload={
            "status": margin_summary.get("status", "unknown"),
            "model_name": "dglm_margin",
            "dispersion_diagnostics": margin_summary.get("dispersion_diagnostics", {}),
        },
    )
    return _task_result(ctx, out, summary=payload, applicability=str(holdout_profile["overall_applicability"]))


is_dglm_margin_primary = _is_dglm_margin_primary
task_dglm_margin_diagnostics = _task_dglm_margin_diagnostics
