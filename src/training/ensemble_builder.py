"""Ensemble assembly policy shared by training and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.models.ensemble_stack import StackingEnsemble
from src.models.ensemble_weighted import compute_weights, spread_stats, weighted_ensemble
from src.training.ensemble_policy import ensemble_component_columns
from src.training.progress import ProgressCallback, emit_progress


def fit_stacker(
    oof: pd.DataFrame,
    *,
    league: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[StackingEnsemble, bool, list[str]]:
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "stacking", "status": "started", "message": "Preparing stacking ensemble"},
    )
    stacker = StackingEnsemble()
    stack_base_cols = ensemble_component_columns(
        [
            c
            for c in [
                "elo_baseline",
                "dynamic_rating",
                "glm_ridge",
                "glm_elastic_net",
                "glm_lasso",
                "gbdt",
                "rf",
                "two_stage",
                "goals_poisson",
                "simulation_first",
                "bayes_bt_state_space",
                "bayes_goals",
                "nn_mlp",
            ]
            if c in oof.columns
        ],
        league=league,
    )
    stack_ready = False
    if not oof.empty and len(stack_base_cols) >= 3:
        stacker.fit(oof.dropna(subset=["home_win"]), base_columns=stack_base_cols, target_col="home_win")
        stack_ready = True
    emit_progress(
        progress_callback,
        {
            "kind": "pipeline",
            "stage": "stacking",
            "status": "completed",
            "message": "Stacking ensemble stage completed",
            "stack_ready": bool(stack_ready),
            "stack_base_count": len(stack_base_cols),
        },
    )
    return stacker, stack_ready, stack_base_cols


def build_oof_metrics(oof: pd.DataFrame) -> list[dict]:
    if oof.empty:
        return []
    y = oof["home_win"].astype(int).to_numpy()
    oof_metrics = []
    for col in [c for c in oof.columns if c not in {"game_id", "home_win", "game_date_utc"}]:
        metrics = metric_bundle(y, oof[col].to_numpy())
        oof_metrics.append(
            {
                "model_name": col,
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "ece": abs(metrics["accuracy"] - y.mean()),
                "calibration_beta": 1.0,
            }
        )
    return oof_metrics


def build_ensemble_outputs(
    upcoming_preds: pd.DataFrame,
    oof_metrics: list[dict],
    stacker: StackingEnsemble,
    stack_ready: bool,
    *,
    league: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    model_cols = [c for c in upcoming_preds.columns if c != "game_id"]
    ensemble_cols = ensemble_component_columns(model_cols, league=league)
    if not ensemble_cols:
        raise RuntimeError("No model predictions were produced for ensemble construction.")

    ensemble_set = set(ensemble_cols)
    filtered_metrics = [row for row in oof_metrics if str(row.get("model_name")) in ensemble_set]
    weights = compute_weights(pd.DataFrame(filtered_metrics))
    if not weights:
        weights = {c: 1.0 for c in ensemble_cols}

    if stack_ready:
        stack_prob = stacker.predict_proba(upcoming_preds)
    else:
        stack_prob = weighted_ensemble(upcoming_preds[ensemble_cols], weights)

    weight_prob = weighted_ensemble(upcoming_preds[ensemble_cols], weights)
    ensemble_prob = np.clip(0.6 * stack_prob + 0.4 * weight_prob, 1e-6, 1 - 1e-6)
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "ensemble", "status": "completed", "message": "Built ensemble probabilities"},
    )
    return ensemble_prob, weights, spread_stats(upcoming_preds, ensemble_cols)
