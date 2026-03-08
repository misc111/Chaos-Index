from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.time import utc_now_iso
from src.common.utils import ensure_dir
from src.evaluation.calibration import calibration_alpha_beta, ece_mce, reliability_table
from src.evaluation.metrics import metric_bundle, per_game_scores
from src.training.cv import time_series_splits
from src.training.train import (
    _fit_suite,
    _predict_suite,
    glm_feature_subset,
    normalize_selected_models,
    select_feature_columns,
)
from src.training.tune import quick_tune_glm



def run_walk_forward_backtest(
    features_df: pd.DataFrame,
    artifacts_dir: str,
    bayes_cfg: dict,
    n_splits: int = 5,
    selected_models: list[str] | None = None,
    selected_feature_columns: list[str] | None = None,
    selected_model_feature_columns: dict[str, list[str]] | None = None,
    allow_nn: bool = False,
    min_train_size: int | None = None,
) -> dict:
    df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc")
    models_selected = normalize_selected_models(selected_models)
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
    glm_cols = glm_feature_subset(feature_cols)
    resolved_min_train_size = min(220, max(80, len(df) // 2)) if min_train_size is None else int(min_train_size)
    splits = time_series_splits(df, n_splits=n_splits, min_train_size=resolved_min_train_size)

    pred_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr = df.loc[tr_idx].copy().sort_values("start_time_utc")
        va = df.loc[va_idx].copy().sort_values("start_time_utc")
        if tr.empty or va.empty:
            continue

        fold_glm_c = 1.0
        fold_glm_cols = (
            selected_model_feature_columns.get("glm_ridge")
            or selected_model_feature_columns.get("glm_logit")
            or glm_cols
        ) if selected_model_feature_columns else glm_cols
        if "glm_ridge" in models_selected:
            tune = quick_tune_glm(
                tr,
                fold_glm_cols,
                n_splits=3,
                min_train_size=min(140, max(70, len(tr) // 2)),
            )
            fold_glm_c = float(tune.get("best_c", 1.0))
        models, _, _, _, _ = _fit_suite(
            tr,
            feature_cols,
            artifacts_dir=artifacts_dir,
            bayes_cfg=bayes_cfg,
            selected_models=models_selected,
            allow_nn=allow_nn,
            glm_feature_cols=fold_glm_cols,
            glm_c=fold_glm_c,
            model_feature_columns=selected_model_feature_columns,
        )
        pred_df, _ = _predict_suite(models, va, feature_cols, selected_models=models_selected)
        pred_df["fold"] = fold
        pred_df["home_win"] = va["home_win"].astype(int).to_numpy()
        pred_df["game_id"] = va["game_id"].to_numpy()
        pred_df["game_date_utc"] = va["game_date_utc"].to_numpy()
        pred_rows.append(pred_df)

    if not pred_rows:
        return {"oof_predictions": pd.DataFrame(), "metrics": pd.DataFrame(), "per_game_scores": pd.DataFrame()}

    oof = pd.concat(pred_rows, ignore_index=True)
    model_cols = [c for c in oof.columns if c not in {"fold", "home_win", "game_id", "game_date_utc"}]

    metrics_rows = []
    score_rows = []
    for col in model_cols:
        model_eval = oof[["game_id", "game_date_utc", "home_win", col]].dropna().copy()
        if model_eval.empty:
            continue
        y = model_eval["home_win"].to_numpy(dtype=int)
        p = model_eval[col].to_numpy(dtype=float)
        m = metric_bundle(y, p)
        c = ece_mce(y, p)
        ab = calibration_alpha_beta(y, p)
        metrics_rows.append({"model_name": col} | m | c | ab)

        s = per_game_scores(y, p)
        s["model_name"] = col
        s["game_id"] = model_eval["game_id"].values
        s["game_date_utc"] = model_eval["game_date_utc"].values
        s["prob_home_win"] = p
        s["outcome_home_win"] = y
        s["as_of_utc"] = utc_now_iso()
        score_rows.append(s)

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values("log_loss")
    scores_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()

    out_dir = ensure_dir(Path(artifacts_dir) / "reports")
    try:
        oof.to_parquet(out_dir / "backtest_oof_predictions.parquet", index=False)
    except Exception:
        oof.to_csv(out_dir / "backtest_oof_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "backtest_metrics.csv", index=False)
    try:
        scores_df.to_parquet(out_dir / "backtest_per_game_scores.parquet", index=False)
    except Exception:
        scores_df.to_csv(out_dir / "backtest_per_game_scores.csv", index=False)

    # Reliability tables
    rel_dir = ensure_dir(Path(artifacts_dir) / "validation" / "backtest" / "reliability")
    for col in metrics_df["model_name"].tolist() if not metrics_df.empty else []:
        model_eval = oof[["home_win", col]].dropna().copy()
        rel = reliability_table(model_eval["home_win"].to_numpy(dtype=int), model_eval[col].to_numpy(dtype=float), n_bins=10)
        rel.to_csv(rel_dir / f"backtest_reliability_{col}.csv", index=False)

    return {
        "oof_predictions": oof,
        "metrics": metrics_df,
        "per_game_scores": scores_df,
    }
