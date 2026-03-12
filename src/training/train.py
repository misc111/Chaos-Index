"""Training orchestration over modular runners and policies.

`train_and_predict` remains the public entry point used by services and tests.
The actual responsibilities now live in dedicated modules so model metadata,
fit logic, prediction logic, ensemble assembly, uncertainty flags, and artifact
writing can evolve independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import ensure_dir, stable_hash
from src.evaluation.metrics import metric_bundle
from src.features.leakage_checks import run_leakage_checks
from src.training.artifact_writer import save_model_artifacts, save_training_outputs
from src.training.ensemble_builder import blend_ensemble_probabilities, build_ensemble_outputs, build_oof_metrics, fit_stacker
from src.training.ensemble_policy import demoted_ensemble_models, ensemble_component_columns
from src.training.feature_selection import (
    glm_feature_subset,
    select_feature_columns,
)
from src.training.fit_runner import fit_model_suite
from src.training.model_catalog import normalize_selected_models
from src.training.penalized_glm import (
    PREFERRED_VALIDATION_PENALIZED_GLM_MODELS,
    resolve_penalized_glm_feature_columns,
    selected_penalized_glm_models,
    tune_penalized_glm_models,
)
from src.training.predict_runner import generate_oof_predictions, predict_model_suite
from src.training.progress import ProgressCallback, emit_progress
from src.training.uncertainty_policy import build_uncertainty_flags

_predict_suite = predict_model_suite
_oof_predictions = generate_oof_predictions


def _fit_suite(*args, **kwargs):
    return fit_model_suite(*args, **kwargs, metric_bundle_fn=metric_bundle)


def train_and_predict(
    features_df: pd.DataFrame,
    feature_set_version: str,
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    selected_feature_columns: list[str] | None = None,
    selected_model_feature_columns: dict[str, list[str]] | None = None,
    league: str | None = None,
) -> dict:
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "prepare_data", "status": "started", "message": "Preparing training datasets"},
    )
    df = features_df.sort_values("start_time_utc").copy()
    train_df = df[df["home_win"].notna()].copy()
    upcoming_df = df[df["home_win"].isna()].copy()
    models_selected = normalize_selected_models(selected_models)
    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "prepare_data",
            "status": "completed",
            "message": "Prepared training and upcoming datasets",
            "n_train": int(len(train_df)),
            "n_upcoming": int(len(upcoming_df)),
            "model_total": len(models_selected),
            "selected_models": models_selected,
        },
    )

    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "feature_selection", "status": "started", "message": "Selecting feature columns"},
    )
    if selected_feature_columns is None:
        feature_cols = select_feature_columns(df)
    else:
        missing_cols = [c for c in selected_feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"selected_feature_columns includes missing columns: {missing_cols}")
        non_numeric = [c for c in selected_feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"selected_feature_columns includes non-numeric columns: {non_numeric}")
        feature_cols = list(selected_feature_columns)
    penalized_glm_feature_cols = resolve_penalized_glm_feature_columns(
        feature_cols,
        selected_models=models_selected,
        model_feature_columns=selected_model_feature_columns,
        fallback_columns=glm_feature_subset(feature_cols),
    )
    glm_cols = (
        penalized_glm_feature_cols.get("glm_ridge")
        or penalized_glm_feature_cols.get("glm_elastic_net")
        or penalized_glm_feature_cols.get("glm_lasso")
        or glm_feature_subset(feature_cols)
    )
    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "feature_selection",
            "status": "completed",
            "message": "Selected feature columns",
            "feature_count": len(feature_cols),
            "glm_feature_count": len(glm_cols),
        },
    )

    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "leakage_checks", "status": "started", "message": "Running leakage checks"},
    )
    issues = run_leakage_checks(df, feature_columns=feature_cols)
    if issues:
        emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "leakage_checks",
                "status": "failed",
                "message": f"Leakage checks failed: {issues}",
            },
        )
        raise RuntimeError(f"Leakage checks failed: {issues}")
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "leakage_checks", "status": "completed", "message": "Leakage checks passed"},
    )

    model_run_prefix = stable_hash({"feature_set_version": feature_set_version, "n_train": len(train_df)})
    model_dir = ensure_dir(Path(artifacts_dir) / "models" / model_run_prefix)

    glm_tune: dict = {"best_c": 1.0, "results": [], "fold_metrics": []}
    glm_best_c = 1.0
    glm_tuning_by_model: dict[str, dict] = {}
    primary_penalized_glm = None
    penalized_models = selected_penalized_glm_models(models_selected)
    if penalized_models:
        emit_progress(
            progress_callback,
            {"kind": "pipeline", "stage": "glm_tuning", "status": "started", "message": "Running GLM hyperparameter tuning"},
        )
        glm_tuning_by_model = tune_penalized_glm_models(
            train_df,
            selected_models=models_selected,
            feature_columns_by_model=penalized_glm_feature_cols,
            n_splits=4,
            min_train_size=min(220, max(100, len(train_df) // 2)),
        )
        primary_penalized_glm = next(
            (model_name for model_name in PREFERRED_VALIDATION_PENALIZED_GLM_MODELS if model_name in glm_tuning_by_model),
            penalized_models[0],
        )
        glm_tune = dict(glm_tuning_by_model.get(primary_penalized_glm, glm_tune))
        glm_best_c = float(glm_tune.get("best_c", 1.0))
        emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "glm_tuning",
                "status": "completed",
                "message": "Completed GLM hyperparameter tuning",
                "glm_best_c": glm_best_c,
                "glm_primary_model": primary_penalized_glm,
                "glm_models_tuned": penalized_models,
            },
        )

    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "fit_models", "status": "started", "message": "Fitting selected models"},
    )
    models, bayes_cols, bayes_diag, nn_included, used_feature_map = fit_model_suite(
        train_df,
        feature_cols,
        artifacts_dir=artifacts_dir,
        bayes_cfg=bayes_cfg,
        selected_models=models_selected,
        progress_callback=progress_callback,
        allow_nn=True,
        glm_feature_cols=glm_cols,
        glm_c=glm_best_c,
        glm_params_by_model=glm_tuning_by_model,
        model_feature_columns=selected_model_feature_columns,
        metric_bundle_fn=metric_bundle,
    )
    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "fit_models",
            "status": "completed",
            "message": "Completed model fitting",
            "fitted_model_count": len(models),
        },
    )
    save_model_artifacts(models, model_dir, progress_callback=progress_callback)

    oof = generate_oof_predictions(
        train_df,
        feature_cols,
        glm_cols,
        artifacts_dir=artifacts_dir,
        bayes_cfg=bayes_cfg,
        selected_models=models_selected,
        progress_callback=progress_callback,
        glm_params_by_model=glm_tuning_by_model,
        model_feature_columns=selected_model_feature_columns,
    )
    stacker, stack_ready, stack_base_cols = fit_stacker(oof, league=league, progress_callback=progress_callback)
    oof_metrics = build_oof_metrics(oof)

    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_upcoming", "status": "started", "message": "Generating upcoming predictions"},
    )
    upcoming_preds, upcoming_extras = predict_model_suite(
        models,
        upcoming_df,
        feature_cols,
        selected_models=models_selected,
        progress_callback=progress_callback,
        phase="predict_upcoming",
    )
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_upcoming", "status": "completed", "message": "Completed upcoming predictions"},
    )
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_train", "status": "started", "message": "Generating in-sample diagnostics"},
    )
    train_preds, _ = predict_model_suite(
        models,
        train_df,
        feature_cols,
        selected_models=models_selected,
        progress_callback=progress_callback,
        phase="predict_train",
    )
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "predict_train", "status": "completed", "message": "Completed in-sample diagnostics"},
    )

    ensemble_prob, weights, spread = build_ensemble_outputs(
        upcoming_preds,
        oof_metrics,
        stacker,
        stack_ready,
        league=league,
        progress_callback=progress_callback,
    )
    historical_oof = oof.copy()
    historical_model_cols = [c for c in historical_oof.columns if c not in {"game_id", "home_win", "game_date_utc"}]
    historical_ensemble_cols = ensemble_component_columns(historical_model_cols, league=league)
    if historical_ensemble_cols:
        historical_oof["ensemble"] = blend_ensemble_probabilities(
            historical_oof,
            historical_ensemble_cols,
            weights,
            stacker,
            stack_ready,
        )

    forecasts = upcoming_df[["game_id", "game_date_utc", "home_team", "away_team", "as_of_utc"]].copy()
    forecasts["ensemble_prob_home_win"] = ensemble_prob
    forecasts["predicted_winner"] = np.where(ensemble_prob >= 0.5, forecasts["home_team"], forecasts["away_team"])
    forecasts = pd.concat([forecasts.reset_index(drop=True), spread.reset_index(drop=True)], axis=1)
    forecasts["bayes_ci_low"] = upcoming_extras.get("bayes_low", np.full(len(forecasts), np.nan))
    forecasts["bayes_ci_high"] = upcoming_extras.get("bayes_high", np.full(len(forecasts), np.nan))
    forecasts["uncertainty_flags_json"] = build_uncertainty_flags(upcoming_df)

    model_cols = [c for c in upcoming_preds.columns if c != "game_id"]
    ensemble_component_cols = ensemble_component_columns(model_cols, league=league)
    per_model_rows = []
    for _, forecast_row in forecasts.iterrows():
        gid = forecast_row["game_id"]
        p_row = upcoming_preds[upcoming_preds["game_id"] == gid].iloc[0]
        per_model = {c: float(p_row[c]) for c in model_cols}
        per_model_rows.append(json.dumps(per_model, sort_keys=True))
    forecasts["per_model_probs_json"] = per_model_rows

    run_payload = {
        "model_run_id": f"run_{model_run_prefix}",
        "league": league,
        "feature_set_version": feature_set_version,
        "selected_models": models_selected,
        "feature_columns": feature_cols,
        "glm_feature_columns": glm_cols,
        "model_feature_columns": used_feature_map,
        "glm_tuning": glm_tune,
        "glm_tuning_by_model": glm_tuning_by_model,
        "glm_primary_model": primary_penalized_glm,
        "glm_best_c": glm_best_c,
        "glm_lasso_best_c": float(glm_tuning_by_model["glm_lasso"]["best_c"]) if "glm_lasso" in glm_tuning_by_model else None,
        "glm_elastic_net_best_c": (
            float(glm_tuning_by_model["glm_elastic_net"]["best_c"]) if "glm_elastic_net" in glm_tuning_by_model else None
        ),
        "glm_elastic_net_best_l1_ratio": (
            float(glm_tuning_by_model["glm_elastic_net"]["best_l1_ratio"])
            if glm_tuning_by_model.get("glm_elastic_net", {}).get("best_l1_ratio") is not None
            else None
        ),
        "bayes_feature_columns": bayes_cols,
        "ensemble_component_columns": ensemble_component_cols,
        "ensemble_demoted_models": [m for m in demoted_ensemble_models(league=league) if m in models_selected],
        "stack_base_columns": stack_base_cols,
        "weights": weights,
        "nn_included": nn_included,
        "bayes_diagnostics": bayes_diag,
        "model_dir": str(model_dir),
    }
    save_training_outputs(
        model_dir,
        forecasts,
        upcoming_preds,
        historical_oof,
        run_payload,
        progress_callback=progress_callback,
    )

    train_metrics = {}
    if not train_preds.empty:
        y = train_df["home_win"].astype(int).to_numpy()
        for col in [c for c in train_preds.columns if c != "game_id"]:
            train_metrics[col] = metric_bundle(y, train_preds[col].to_numpy())

    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "train_complete",
            "status": "completed",
            "message": "Model training completed",
            "model_run_id": run_payload["model_run_id"],
        },
    )
    return {
        "model_run_id": run_payload["model_run_id"],
        "feature_columns": feature_cols,
        "weights": weights,
        "oof_metrics": oof_metrics,
        "oof_predictions": historical_oof,
        "forecasts": forecasts,
        "upcoming_model_probs": upcoming_preds,
        "train_metrics": train_metrics,
        "stack_ready": stack_ready,
        "model_dir": str(model_dir),
        "run_payload": run_payload,
        "models": models,
        "train_df": train_df,
        "upcoming_df": upcoming_df,
    }
