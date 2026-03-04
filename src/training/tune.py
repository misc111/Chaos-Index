from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.models.glm_logit import GLMLogitModel
from src.training.cv import time_series_splits



def quick_tune_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    c_grid: list[float] | None = None,
    n_splits: int = 4,
    min_train_size: int = 220,
) -> dict:
    if c_grid is None:
        c_grid = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    train = df[df["home_win"].notna()].copy().sort_values("start_time_utc")
    if len(train) < max(80, min_train_size + 20):
        return {"best_c": 1.0, "results": [], "fold_metrics": []}

    splits = time_series_splits(train, n_splits=n_splits, min_train_size=min_train_size)
    if not splits:
        return {"best_c": 1.0, "results": [], "fold_metrics": []}

    rows = []
    fold_rows = []
    for c in c_grid:
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = train.loc[tr_idx].copy().sort_values("start_time_utc")
            va = train.loc[va_idx].copy().sort_values("start_time_utc")
            if tr.empty or va.empty:
                continue
            model = GLMLogitModel(c=float(c))
            model.fit(tr, feature_cols)
            p = model.predict_proba(va)
            met = metric_bundle(va["home_win"].astype(int).to_numpy(), p)
            fold_scores.append(met)
            fold_rows.append(
                {
                    "c": float(c),
                    "fold": int(fold),
                    "n_train": int(len(tr)),
                    "n_valid": int(len(va)),
                    "log_loss": float(met["log_loss"]),
                    "brier": float(met["brier"]),
                    "accuracy": float(met["accuracy"]),
                    "auc": float(met["auc"]),
                }
            )
        if not fold_scores:
            continue
        rows.append(
            {
                "c": float(c),
                "folds_used": int(len(fold_scores)),
                "log_loss": float(pd.DataFrame(fold_scores)["log_loss"].mean()),
                "brier": float(pd.DataFrame(fold_scores)["brier"].mean()),
                "accuracy": float(pd.DataFrame(fold_scores)["accuracy"].mean()),
                "auc": float(pd.DataFrame(fold_scores)["auc"].mean()),
            }
        )

    if not rows:
        return {"best_c": 1.0, "results": [], "fold_metrics": fold_rows}

    best = sorted(rows, key=lambda r: (r["log_loss"], r["brier"], -r["accuracy"]))[0]
    return {"best_c": float(best["c"]), "results": rows, "fold_metrics": fold_rows}
