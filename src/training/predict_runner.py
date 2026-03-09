"""Prediction and OOF runners separated from orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.simulation.game_simulator import GameSimulator
from src.training.cv import time_series_splits
from src.training.fit_runner import fit_model_suite
from src.training.penalized_glm import resolve_penalized_glm_feature_columns, selected_penalized_glm_models, tune_penalized_glm_models
from src.training.progress import ProgressCallback, emit_progress


def predict_model_suite(
    models: dict[str, object],
    df: pd.DataFrame,
    feature_cols: list[str],
    selected_models: list[str],
    progress_callback: ProgressCallback | None = None,
    phase: str = "predict",
) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame({"game_id": df["game_id"].values})
    extras: dict = {}
    selected = set(selected_models)

    if "elo_baseline" in selected:
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "elo_baseline",
                "stage": phase,
                "status": "started",
                "message": f"Running {phase} for elo_baseline",
            },
        )
        out["elo_baseline"] = np.clip(df.get("elo_home_prob", 0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "elo_baseline",
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for elo_baseline",
            },
        )
    if "dynamic_rating" in selected:
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "dynamic_rating",
                "stage": phase,
                "status": "started",
                "message": f"Running {phase} for dynamic_rating",
            },
        )
        out["dynamic_rating"] = np.clip(df.get("dyn_home_prob", 0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "dynamic_rating",
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for dynamic_rating",
            },
        )

    for name, model in models.items():
        if name not in selected:
            continue
        emit_progress(
            progress_callback,
            {"kind": "model", "model": name, "stage": phase, "status": "started", "message": f"Running {phase} for {name}"},
        )
        if name == "goals_poisson":
            out[name] = model.predict_proba(df)
        elif name == "bayes_goals":
            mean, low, high = model.predict_proba(df)
            out[name] = mean
            extras["bayes_goals_low"] = low
            extras["bayes_goals_high"] = high
        elif name == "bayes_bt_state_space":
            summary = model.predict_summary(df)
            out[name] = summary.mean
            extras["bayes_low"] = summary.low
            extras["bayes_high"] = summary.high
            extras["bayes_pred_var"] = summary.pred_var
        else:
            out[name] = model.predict_proba(df)
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": name,
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for {name}",
            },
        )

    if "simulation_first" in selected:
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "simulation_first",
                "stage": phase,
                "status": "started",
                "message": f"Running {phase} for simulation_first",
            },
        )
        sim = GameSimulator(seed=42)
        sim_df = sim.simulate_dataframe(df, n_sims=3500)
        out = out.merge(sim_df[["game_id", "sim_prob_home_win"]], on="game_id", how="left")
        out = out.rename(columns={"sim_prob_home_win": "simulation_first"})
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": "simulation_first",
                "stage": phase,
                "status": "completed",
                "message": f"Completed {phase} for simulation_first",
            },
        )

    return out, extras


def generate_oof_predictions(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    glm_feature_cols: list[str],
    artifacts_dir: str,
    bayes_cfg: dict,
    selected_models: list[str],
    progress_callback: ProgressCallback | None = None,
    glm_params_by_model: dict[str, dict] | None = None,
    model_feature_columns: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    splits = time_series_splits(train_df, n_splits=5, min_train_size=min(220, max(80, len(train_df) // 2)))
    rows = []
    selected = set(selected_models)
    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "oof_validation",
            "status": "started",
            "message": "Starting time-series OOF validation",
            "fold_total": len(splits),
        },
    )

    for fold_number, (tr_idx, va_idx) in enumerate(splits, start=1):
        emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "oof_fold",
                "status": "started",
                "message": f"Starting OOF fold {fold_number}/{len(splits)}",
                "fold": fold_number,
                "fold_total": len(splits),
            },
        )
        tr = train_df.loc[tr_idx].copy().sort_values("start_time_utc")
        va = train_df.loc[va_idx].copy().sort_values("start_time_utc")
        if tr.empty or va.empty:
            emit_progress(
                progress_callback,
                {
                    "kind": "pipeline",
                    "stage": "oof_fold",
                    "status": "skipped",
                    "message": f"Skipped OOF fold {fold_number}/{len(splits)} because split was empty",
                    "fold": fold_number,
                    "fold_total": len(splits),
                },
            )
            continue
        fold_glm_tuning = dict(glm_params_by_model or {})
        if selected_penalized_glm_models(selected_models):
            fold_glm_cols_by_model = resolve_penalized_glm_feature_columns(
                feature_cols,
                selected_models=selected_models,
                model_feature_columns=model_feature_columns,
                fallback_columns=glm_feature_cols,
            )
            fold_glm_tuning = tune_penalized_glm_models(
                tr,
                selected_models=selected_models,
                feature_columns_by_model=fold_glm_cols_by_model,
                n_splits=3,
                min_train_size=min(140, max(70, len(tr) // 2)),
            )
        models, _, _, _, _ = fit_model_suite(
            tr,
            feature_cols,
            artifacts_dir=artifacts_dir,
            bayes_cfg=bayes_cfg,
            selected_models=selected_models,
            progress_callback=progress_callback,
            allow_nn=False,
            glm_feature_cols=glm_feature_cols,
            glm_params_by_model=fold_glm_tuning,
            model_feature_columns=model_feature_columns,
        )
        pred, _ = predict_model_suite(
            models,
            va,
            feature_cols,
            selected_models=selected_models,
            progress_callback=progress_callback,
            phase="oof_predict",
        )
        pred["home_win"] = va["home_win"].to_numpy()
        pred["game_date_utc"] = va["game_date_utc"].to_numpy()
        rows.append(pred)
        emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "oof_fold",
                "status": "completed",
                "message": f"Completed OOF fold {fold_number}/{len(splits)}",
                "fold": fold_number,
                "fold_total": len(splits),
            },
        )

    if not rows:
        emit_progress(
            progress_callback,
            {
                "kind": "pipeline",
                "stage": "oof_validation",
                "status": "completed",
                "message": "OOF validation completed with no folds",
                "fold_total": len(splits),
            },
        )
        return pd.DataFrame()
    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "oof_validation",
            "status": "completed",
            "message": "Completed time-series OOF validation",
            "fold_total": len(splits),
        },
    )
    return pd.concat(rows, ignore_index=True)
