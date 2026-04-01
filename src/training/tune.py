from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.models.glm_penalized import PENALIZED_GLM_MODEL_NAMES, build_penalized_glm, penalized_glm_config
from src.training.cv import time_series_splits
from src.training.lambda_search import c_to_lambda, default_l1_ratio_grid, default_lambda_grid, lambda_to_c


def _default_c_grid(model_name: str) -> list[float]:
    return [lambda_to_c(value) for value in default_lambda_grid(model_name)]


def _default_tune_result(model_name: str) -> dict:
    config = penalized_glm_config(model_name)
    best_lambda = c_to_lambda(float(config.default_c))
    best_params = {"lambda": best_lambda, "c": float(config.default_c)}
    if config.default_l1_ratio is not None:
        best_params["l1_ratio"] = float(config.default_l1_ratio)
    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_lambda": best_lambda,
        "best_c": float(config.default_c),
        "best_l1_ratio": None if config.default_l1_ratio is None else float(config.default_l1_ratio),
        "results": [],
        "fold_metrics": [],
    }


def quick_tune_penalized_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_name: str = "glm_ridge",
    c_grid: list[float] | None = None,
    l1_ratio_grid: list[float] | None = None,
    n_splits: int = 4,
    min_train_size: int = 220,
) -> dict:
    if model_name not in PENALIZED_GLM_MODEL_NAMES:
        raise ValueError(f"Unsupported penalized GLM '{model_name}'. Valid={list(PENALIZED_GLM_MODEL_NAMES)}")

    config = penalized_glm_config(model_name)
    c_values = list(c_grid) if c_grid is not None else _default_c_grid(model_name)
    l1_ratio_values = list(l1_ratio_grid) if l1_ratio_grid is not None else default_l1_ratio_grid(model_name)
    default_result = _default_tune_result(model_name)

    train = df[df["home_win"].notna()].copy().sort_values("start_time_utc")
    if len(train) < max(80, min_train_size + 20):
        return default_result

    splits = time_series_splits(train, n_splits=n_splits, min_train_size=min_train_size)
    if not splits:
        return default_result

    if config.penalty == "elasticnet":
        param_grid = [
            {
                "lambda": c_to_lambda(float(c)),
                "c": float(c),
                "l1_ratio": float(l1_ratio),
            }
            for c in c_values
            for l1_ratio in l1_ratio_values
        ]
    else:
        param_grid = [{"lambda": c_to_lambda(float(c)), "c": float(c)} for c in c_values]

    rows = []
    fold_rows = []
    for params in param_grid:
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = train.loc[tr_idx].copy().sort_values("start_time_utc")
            va = train.loc[va_idx].copy().sort_values("start_time_utc")
            if tr.empty or va.empty:
                continue
            model = build_penalized_glm(
                model_name,
                c=float(params["c"]),
                l1_ratio=params.get("l1_ratio"),
            )
            model.fit(tr, feature_cols)
            p = model.predict_proba(va)
            met = metric_bundle(va["home_win"].astype(int).to_numpy(), p)
            fold_scores.append(met)
            fold_rows.append(
                {
                    "model_name": model_name,
                    **params,
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
                "model_name": model_name,
                **params,
                "folds_used": int(len(fold_scores)),
                "log_loss": float(pd.DataFrame(fold_scores)["log_loss"].mean()),
                "brier": float(pd.DataFrame(fold_scores)["brier"].mean()),
                "accuracy": float(pd.DataFrame(fold_scores)["accuracy"].mean()),
                "auc": float(pd.DataFrame(fold_scores)["auc"].mean()),
            }
        )

    if not rows:
        out = dict(default_result)
        out["fold_metrics"] = fold_rows
        return out

    best = sorted(rows, key=lambda r: (r["log_loss"], r["brier"], -r["accuracy"]))[0]
    best_params = {"lambda": float(best["lambda"]), "c": float(best["c"])}
    best_l1_ratio = None
    if "l1_ratio" in best:
        best_l1_ratio = float(best["l1_ratio"])
        best_params["l1_ratio"] = best_l1_ratio

    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_lambda": float(best["lambda"]),
        "best_c": float(best["c"]),
        "best_l1_ratio": best_l1_ratio,
        "results": rows,
        "fold_metrics": fold_rows,
    }


def quick_tune_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    c_grid: list[float] | None = None,
    n_splits: int = 4,
    min_train_size: int = 220,
) -> dict:
    return quick_tune_penalized_glm(
        df,
        feature_cols,
        model_name="glm_ridge",
        c_grid=c_grid,
        n_splits=n_splits,
        min_train_size=min_train_size,
    )
