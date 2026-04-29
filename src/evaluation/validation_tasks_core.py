"""Core validation tasks that do not own probability-market reporting."""

from __future__ import annotations

from src.common.utils import ensure_dir
from src.evaluation.diagnostics_glm import save_glm_diagnostics
from src.evaluation.diagnostics_ml import permutation_importance_report
from src.evaluation.mlb_data_quality import assess_mlb_data_quality
from src.evaluation.validation_artifacts import ValidationOutputs, validation_path as _validation_path
from src.evaluation.validation_context import ValidationContext, date_bounds as _date_bounds, holdout_df as _holdout_df
from src.evaluation.validation_nonlinearity import assess_nonlinearity
from src.evaluation.validation_stability import assess_multicollinearity
from src.evaluation.validation_task_support import ValidationTaskResult, _task_result

def _task_split_summary(ctx: ValidationContext) -> ValidationTaskResult:
    train_start, train_end = _date_bounds(ctx.tr)
    valid_start, valid_end = _date_bounds(ctx.va)
    holdout = _holdout_df(ctx)
    holdout_start, holdout_end = _date_bounds(holdout)
    payload = {
        "status": "ok",
        "split_mode": ctx.split_plan.resolved_mode,
        "split_method": ctx.split_plan.resolved_method,
        "requested_split_mode": ctx.split_plan.requested_mode,
        "requested_split_method": ctx.split_plan.requested_method,
        "train_fraction": ctx.split_plan.train_fraction,
        "validation_fraction": ctx.split_plan.validation_fraction,
        "holdout_fraction": ctx.split_plan.holdout_fraction,
        "random_seed": ctx.split_plan.random_seed,
        "split_note": ctx.split_plan.note,
        "n_train": int(len(ctx.tr)),
        "n_validation": int(len(ctx.va)),
        "n_holdout": int(len(holdout)),
        "n_fit": int(len(ctx.fit_df)),
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": valid_start,
        "validation_end": valid_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
    }
    out = ValidationOutputs()
    out.add_json(section="split_summary", file_name=_validation_path("split", "validation_split_summary.json"), payload=payload)
    return _task_result(ctx, out, summary=payload)


def _task_glm_diagnostics(ctx: ValidationContext) -> ValidationTaskResult:
    if ctx.glm is None:
        return ValidationTaskResult()

    diagnostic_df = ctx.fit_df if not ctx.fit_df.empty else (ctx.tr if not ctx.tr.empty else ctx.train_df)
    report = save_glm_diagnostics(
        diagnostic_df,
        glm=ctx.glm,
        target_col="home_win",
        out_dir=str(ensure_dir(ctx.out_dir / "glm" / "residuals" / "plots")),
        prefix="glm_validation",
        relative_plot_dir=_validation_path("glm", "residuals", "plots"),
    )
    out = ValidationOutputs()
    out.add_json(
        section="glm_residual_summary",
        file_name=_validation_path("glm", "residuals", "validation_glm_residual_summary.json"),
        payload=report["summary"],
    )
    out.add_csv(
        section="glm_residual_feature_summary",
        file_name=_validation_path("glm", "residuals", "validation_glm_residual_feature_summary.csv"),
        rows=report["feature_summary"],
    )
    out.add_csv(
        section="glm_working_residual_bins_linear_predictor",
        file_name=_validation_path("glm", "residuals", "validation_glm_working_residual_bins_linear_predictor.csv"),
        rows=report["linear_predictor_bins"],
    )
    out.add_csv(
        section="glm_working_residual_bins_features",
        file_name=_validation_path("glm", "residuals", "validation_glm_working_residual_bins_features.csv"),
        rows=report["feature_working_bins"],
    )
    out.add_csv(
        section="glm_working_residual_bins_weight",
        file_name=_validation_path("glm", "residuals", "validation_glm_working_residual_bins_weight.csv"),
        rows=report["weight_bins"],
    )
    out.add_csv(
        section="glm_partial_residual_bins",
        file_name=_validation_path("glm", "residuals", "validation_glm_partial_residual_bins.csv"),
        rows=report["partial_residual_bins"],
    )
    out.add_csv(
        section="glm_raw_residual_rows",
        file_name=_validation_path("glm", "residuals", "validation_glm_raw_residual_rows.csv"),
        rows=report["raw_residuals"],
    )
    out.add_csv(
        section="glm_residual_vs_fitted_bins",
        file_name=_validation_path("glm", "residuals", "validation_glm_residual_vs_fitted_bins.csv"),
        rows=report["residual_vs_fitted_bins"],
    )
    out.add_csv(
        section="glm_grouped_residual_summary",
        file_name=_validation_path("glm", "residuals", "validation_glm_grouped_residual_summary.csv"),
        rows=report["grouped_residual_summary"],
    )
    out.add_json(
        section="glm_grouped_residual_metadata",
        file_name=_validation_path("glm", "residuals", "validation_glm_grouped_residual_metadata.json"),
        payload=report["grouped_residual_metadata"],
    )
    return _task_result(ctx, out, summary=report["summary"])


def _task_permutation_importance(ctx: ValidationContext) -> ValidationOutputs:
    holdout = _holdout_df(ctx)
    if holdout.empty:
        return ValidationOutputs()

    for model_name in ["gbdt", "rf"]:
        model = ctx.models.get(model_name)
        if model is None or len(holdout) <= 25:
            continue
        model_feature_cols = getattr(model, "feature_columns", ctx.feature_cols) or ctx.feature_cols
        permutation_importance_report(
            model.model,
            holdout[model_feature_cols],
            holdout["home_win"].astype(int).to_numpy(),
            out_dir=str(ensure_dir(ctx.out_dir / "diagnostics" / "permutation_importance")),
            model_name=model_name,
        )
    return ValidationOutputs()


def _task_collinearity(ctx: ValidationContext) -> ValidationTaskResult:
    report = assess_multicollinearity(ctx.train_df, features=ctx.diagnostic_feature_cols)
    out = ValidationOutputs()
    out.add_json(
        section="collinearity_summary",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_summary.json"),
        payload=report["summary"],
    )
    out.add_csv(
        section="vif",
        file_name=_validation_path("diagnostics", "collinearity", "validation_vif.csv"),
        rows=report["vif"],
    )
    out.add_csv(
        section="collinearity_structural",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_structural.csv"),
        rows=report["structural"],
    )
    out.add_csv(
        section="collinearity_pairs",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_pairs.csv"),
        rows=report["pairwise"],
    )
    out.add_csv(
        section="collinearity_condition",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_condition.csv"),
        rows=report["condition"],
    )
    out.add_csv(
        section="collinearity_variance_decomposition",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_variance_decomposition.csv"),
        rows=report["variance_decomposition"],
    )
    return _task_result(ctx, out, summary=report["summary"])


def _task_nonlinearity(ctx: ValidationContext) -> ValidationTaskResult:
    report = assess_nonlinearity(
        ctx.tr if not ctx.tr.empty else ctx.train_df,
        ctx.va if not ctx.va.empty else (_holdout_df(ctx) if not _holdout_df(ctx).empty else ctx.train_df),
        features=ctx.diagnostic_feature_cols,
    )
    out = ValidationOutputs()
    out.add_json(
        section="nonlinearity_summary",
        file_name=_validation_path("diagnostics", "nonlinearity", "validation_nonlinearity_summary.json"),
        payload=report["summary"],
    )
    out.add_csv(
        section="nonlinearity_feature_summary",
        file_name=_validation_path("diagnostics", "nonlinearity", "validation_nonlinearity_feature_summary.csv"),
        rows=report["feature_summary"],
    )
    out.add_csv(
        section="nonlinearity_curve_points",
        file_name=_validation_path("diagnostics", "nonlinearity", "validation_nonlinearity_curve_points.csv"),
        rows=report["curve_points"],
    )
    return _task_result(ctx, out, summary=report["summary"])

def _task_mlb_data_quality(ctx: ValidationContext) -> ValidationTaskResult:
    summary, issues = assess_mlb_data_quality(ctx.cfg.paths.db_path)
    out = ValidationOutputs()
    out.add_json(
        section="mlb_data_quality_summary",
        file_name=_validation_path("diagnostics", "data_quality", "validation_mlb_data_quality_summary.json"),
        payload=summary,
    )
    out.add_csv(
        section="mlb_data_quality_issues",
        file_name=_validation_path("diagnostics", "data_quality", "validation_mlb_data_quality_issues.csv"),
        rows=issues,
    )
    applicability = "partial" if int(summary.get("n_pending_checks", 0)) > 0 else "applicable"
    return _task_result(ctx, out, summary=summary, applicability=applicability)
