"""Probability, calibration, classification, and market-truth validation tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.market_truth import build_market_truth, load_market_truth_moneyline_odds
from src.evaluation.validation_artifacts import ValidationOutputs, validation_path as _validation_path
from src.evaluation.validation_classification import validate_logistic_probability_model
from src.evaluation.validation_context import ValidationContext, holdout_df as _holdout_df
from src.evaluation.validation_task_support import (
    ValidationTaskResult,
    _binary_holdout_profile,
    _primary_model_key,
    _probability_model_family_applicability,
    _task_result,
    _unsupported_task_result,
)

def _task_calibration(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    p = ctx.glm.predict_proba(holdout)
    y = holdout["home_win"].astype(int).to_numpy()
    holdout_profile = _binary_holdout_profile(y)
    metric_applicability = {
        "ece_mce": "applicable" if holdout_profile["n_obs"] > 0 else "not_applicable",
        "alpha_beta": "applicable" if holdout_profile["has_class_contrast"] else "not_applicable",
        "brier_decomposition": "applicable" if holdout_profile["n_obs"] > 0 else "not_applicable",
    }
    payload = calibration_alpha_beta(y, p) | ece_mce(y, p)
    payload |= brier_decompose(y, p)
    payload["holdout_profile"] = holdout_profile
    payload["metric_applicability"] = metric_applicability
    payload["model_family_applicability"] = _probability_model_family_applicability(
        ctx,
        task_name="calibration",
        requires_class_contrast=False,
        class_contrast_available=bool(holdout_profile["has_class_contrast"]),
        n_obs=int(holdout_profile["n_obs"]),
    )

    out = ValidationOutputs()
    out.add_json(
        section="calibration_robustness",
        file_name=_validation_path("diagnostics", "calibration", "validation_calibration_robustness.json"),
        payload=payload,
    )
    return _task_result(ctx, out, summary=payload, applicability=str(holdout_profile["overall_applicability"]))


def _task_classification_curves(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    p = ctx.glm.predict_proba(holdout)
    y = holdout["home_win"].astype(int).to_numpy()
    report = validate_logistic_probability_model(
        y,
        p,
        bins=max(2, int(ctx.cfg.modeling.calibration_bins)),
        current_tossup_half_width=0.05,
        plot_dir=ctx.plots_dir,
        plot_prefix="",
    )
    holdout_profile = dict(report["applicability_summary"])
    metric_applicability = {
        "quantile_curve": str(holdout_profile["calibration_curve_applicability"]),
        "actual_vs_predicted_curve": str(holdout_profile["calibration_curve_applicability"]),
        "lift_curve": str(holdout_profile["lift_curve_applicability"]),
        "lorenz_curve": str(holdout_profile["lift_curve_applicability"]),
        "roc_curve": str(holdout_profile["roc_curve_applicability"]),
        "operating_points": str(holdout_profile["roc_curve_applicability"]),
        "tossup_sweep": str(holdout_profile["calibration_curve_applicability"]),
    }
    model_family_applicability = _probability_model_family_applicability(
        ctx,
        task_name="classification_curves",
        requires_class_contrast=True,
        class_contrast_available=bool(holdout_profile["has_class_contrast"]),
        n_obs=int(holdout_profile["n_obs"]),
    )

    out = ValidationOutputs()
    out.add_json(
        section="logit_quantile_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_quantile_summary.json"),
        payload=report["quantile_summary"],
    )
    out.add_csv(
        section="logit_quantile_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_quantile_curve.csv"),
        rows=report["quantile_curve"],
    )
    out.add_json(
        section="logit_actual_vs_predicted_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_actual_vs_predicted_summary.json"),
        payload=report["actual_vs_predicted_summary"],
    )
    out.add_csv(
        section="logit_actual_vs_predicted_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_actual_vs_predicted_curve.csv"),
        rows=report["actual_vs_predicted_curve"],
    )
    out.add_json(
        section="logit_lift_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lift_summary.json"),
        payload=report["lift_summary"],
    )
    out.add_csv(
        section="logit_lift_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lift_curve.csv"),
        rows=report["lift_curve"],
    )
    out.add_json(
        section="logit_lorenz_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lorenz_summary.json"),
        payload=report["lorenz_summary"],
    )
    out.add_csv(
        section="logit_lorenz_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lorenz_curve.csv"),
        rows=report["lorenz_curve"],
    )
    out.add_json(
        section="logit_roc_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_roc_summary.json"),
        payload=report["roc_summary"],
    )
    out.add_csv(
        section="logit_roc_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_roc_curve.csv"),
        rows=report["roc_curve"],
    )
    out.add_csv(
        section="logit_operating_points",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_operating_points.csv"),
        rows=report["operating_points"],
    )
    out.add_json(
        section="logit_tossup_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_tossup_summary.json"),
        payload=report["tossup_summary"],
    )
    out.add_csv(
        section="logit_tossup_sweep",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_tossup_sweep.csv"),
        rows=report["tossup_sweep"],
    )
    classification_summary = {
        "holdout_profile": holdout_profile,
        "metric_applicability": metric_applicability,
        "model_family_applicability": model_family_applicability,
        "quantile_summary": report["quantile_summary"],
        "actual_vs_predicted_summary": report["actual_vs_predicted_summary"],
        "lift_summary": report["lift_summary"],
        "lorenz_summary": report["lorenz_summary"],
        "roc_summary": report["roc_summary"],
        "tossup_summary": report["tossup_summary"],
    }
    out.add_json(
        section="logit_classification_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_classification_summary.json"),
        payload=classification_summary,
    )
    summary_payload = dict(report["roc_summary"])
    summary_payload["quantile_summary"] = report["quantile_summary"]
    summary_payload["actual_vs_predicted_summary"] = report["actual_vs_predicted_summary"]
    summary_payload["lift_summary"] = report["lift_summary"]
    summary_payload["lorenz_summary"] = report["lorenz_summary"]
    summary_payload["tossup_summary"] = report["tossup_summary"]
    summary_payload["holdout_profile"] = holdout_profile
    summary_payload["metric_applicability"] = metric_applicability
    summary_payload["model_family_applicability"] = model_family_applicability
    return _task_result(
        ctx,
        out,
        summary=summary_payload,
        applicability=str(holdout_profile["overall_applicability"]),
    )


def _validation_market_truth_prediction_frame(ctx: ValidationContext, holdout: pd.DataFrame) -> pd.DataFrame:
    meta_columns = [
        column
        for column in (
            "game_id",
            "season",
            "game_date_utc",
            "start_time_utc",
            "available_as_of_utc",
            "home_team",
            "away_team",
            "home_win",
        )
        if column in holdout.columns
    ]
    predictions = holdout[meta_columns].copy()
    start = pd.to_datetime(predictions["start_time_utc"], errors="coerce", utc=True)
    if "available_as_of_utc" in predictions.columns:
        available = pd.to_datetime(predictions["available_as_of_utc"], errors="coerce", utc=True)
    else:
        available = pd.Series(pd.NaT, index=predictions.index, dtype="datetime64[ns, UTC]")
    prediction_as_of = available.where(available.notna() & start.notna() & available.le(start), start - pd.Timedelta(hours=1))
    predictions["as_of_utc"] = prediction_as_of.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(prediction_as_of.notna(), None)
    model_key = _primary_model_key(ctx) or "primary_model"
    predictions["model_name"] = model_key
    predictions["prob_home_win"] = np.asarray(ctx.glm.predict_proba(holdout), dtype=float)
    return predictions


def _task_market_truth(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    required = {"game_id", "start_time_utc", "home_win"}
    missing = sorted(required - set(holdout.columns))
    if ctx.glm is None:
        return _unsupported_task_result(
            ctx,
            task_name="market_truth",
            note="Market-truth validation requires a primary probability model.",
        )
    if holdout.empty or missing:
        return _unsupported_task_result(
            ctx,
            task_name="market_truth",
            note=f"Market-truth validation requires holdout rows with {sorted(required)}. Missing={missing}",
        )

    game_ids = [int(value) for value in holdout["game_id"].dropna().unique().tolist()]
    odds_lines = load_market_truth_moneyline_odds(ctx.cfg.paths.db_path, league=ctx.league, game_ids=game_ids)
    if odds_lines.empty:
        return _unsupported_task_result(
            ctx,
            task_name="market_truth",
            note="No moneyline odds snapshots were available for the validation holdout games.",
        )

    predictions = _validation_market_truth_prediction_frame(ctx, holdout)
    model_key = _primary_model_key(ctx) or "primary_model"
    result = build_market_truth(
        predictions,
        odds_lines,
        model_names=[model_key],
        min_edge=0.0,
        calibration_bins=ctx.cfg.modeling.calibration_bins,
    )
    integrity_counts = (
        result.per_game["timestamp_integrity"].value_counts(dropna=False).to_dict() if "timestamp_integrity" in result.per_game.columns else {}
    )
    summary_rows = result.summary.to_dict(orient="records") if not result.summary.empty else []
    current_coverage = float(result.summary["current_market_coverage"].min()) if "current_market_coverage" in result.summary else 0.0
    closing_coverage = float(result.summary["closing_market_coverage"].min()) if "closing_market_coverage" in result.summary else 0.0
    applicability = "applicable" if current_coverage > 0 and closing_coverage > 0 else "partial"
    payload = {
        "status": "ok" if summary_rows else "partial",
        "market_truth_classification": "betting_overlay",
        "note": "Market truth compares validation probabilities against prediction-time and closing no-vig moneyline prices.",
        "n_predictions": int(len(result.per_game)),
        "timestamp_integrity_counts": {str(key): int(value) for key, value in integrity_counts.items()},
        "models": summary_rows,
        "market_data_coverage": {
            "current_market_min": current_coverage,
            "closing_market_min": closing_coverage,
        },
        "model_family_applicability": _probability_model_family_applicability(
            ctx,
            task_name="market_truth",
            requires_class_contrast=False,
            class_contrast_available=True,
            n_obs=int(len(result.per_game)),
        ),
    }

    out = ValidationOutputs()
    out.add_json(
        section="market_truth_summary",
        file_name=_validation_path("market_truth", "validation_market_truth_summary.json"),
        payload=payload,
    )
    out.add_csv(
        section="market_truth_by_model",
        file_name=_validation_path("market_truth", "validation_market_truth_by_model.csv"),
        rows=result.per_game,
    )
    out.add_csv(
        section="market_truth_calibration",
        file_name=_validation_path("market_truth", "validation_market_truth_calibration.csv"),
        rows=result.calibration,
    )
    return _task_result(ctx, out, summary=payload, applicability=applicability)
