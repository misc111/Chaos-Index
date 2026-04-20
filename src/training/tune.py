from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.evaluation.metrics import metric_bundle
from src.models.glm_penalized import PENALIZED_GLM_MODEL_NAMES, build_penalized_glm, penalized_glm_config
from src.training.cv import time_series_splits
from src.training.lambda_search import (
    default_l1_ratio_grid,
    default_lambda_grid,
    lambda_to_c,
    penalty_choice,
)


def _default_c_grid(model_name: str) -> list[float]:
    return [lambda_to_c(value) for value in default_lambda_grid(model_name)]


def _fit_summary(model: object | None, *, top_n: int = 8) -> dict[str, Any]:
    fit_summary_fn = getattr(model, "fit_metadata_dict", None)
    if not callable(fit_summary_fn):
        return {}
    try:
        summary = fit_summary_fn(top_n=top_n)
    except TypeError:
        summary = fit_summary_fn()
    return dict(summary) if isinstance(summary, dict) else {}


def _leading_features(summary: dict[str, Any], *, top_n: int = 5) -> list[str]:
    features: list[str] = []
    for row in summary.get("active_coefficient_summary", []):
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "").strip()
        if feature:
            features.append(feature)
    return features[:top_n]


def _aggregate_fold_fit_summaries(fit_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    active_counts: list[int] = []
    feature_counter: Counter[str] = Counter()
    coefficient_stats: dict[str, dict[str, Any]] = {}
    coefficient_columns: list[str] = []

    for summary in fit_summaries:
        count = summary.get("active_parameter_count")
        if isinstance(count, (int, float)) and pd.notna(count):
            active_counts.append(int(count))

        for feature in summary.get("active_features", []):
            token = str(feature).strip()
            if token:
                feature_counter[token] += 1

        if not coefficient_columns:
            coefficient_columns = [
                str(column)
                for column in summary.get("coefficient_columns", [])
                if str(column).strip()
            ]

        for row in summary.get("active_coefficient_summary", []):
            if not isinstance(row, dict):
                continue
            feature = str(row.get("feature") or "").strip()
            if not feature:
                continue
            stats = coefficient_stats.setdefault(
                feature,
                {
                    "fold_count": 0,
                    "sum_abs_coef_scaled": 0.0,
                    "sum_abs_coef_original": 0.0,
                    "n_abs_coef_original": 0,
                    "signs": set(),
                },
            )
            stats["fold_count"] += 1
            stats["sum_abs_coef_scaled"] += float(row.get("abs_coef_scaled", 0.0) or 0.0)
            abs_original = row.get("abs_coef_original")
            if isinstance(abs_original, (int, float)) and pd.notna(abs_original):
                stats["sum_abs_coef_original"] += float(abs_original)
                stats["n_abs_coef_original"] += 1
            sign = row.get("sign")
            if isinstance(sign, (int, float)) and pd.notna(sign):
                stats["signs"].add(int(sign))

    leading_coefficients: list[dict[str, Any]] = []
    for feature, stats in coefficient_stats.items():
        n_original = int(stats["n_abs_coef_original"])
        leading_coefficients.append(
            {
                "feature": feature,
                "fold_count": int(stats["fold_count"]),
                "mean_abs_coef_scaled": float(stats["sum_abs_coef_scaled"] / max(int(stats["fold_count"]), 1)),
                "mean_abs_coef_original": (
                    float(stats["sum_abs_coef_original"] / n_original) if n_original > 0 else None
                ),
                "sign_consistent": len(stats["signs"]) <= 1,
            }
        )

    leading_coefficients.sort(
        key=lambda row: (-int(row["fold_count"]), -float(row["mean_abs_coef_scaled"]), str(row["feature"]))
    )
    feature_frequency = [
        {"feature": feature, "fold_count": int(count)}
        for feature, count in feature_counter.most_common(8)
    ]

    if not active_counts:
        return {
            "active_parameter_count_mean": None,
            "active_parameter_count_min": None,
            "active_parameter_count_max": None,
            "active_feature_union": [],
            "active_feature_frequency": feature_frequency,
            "leading_coefficients": leading_coefficients[:8],
            "coefficient_columns": coefficient_columns,
        }

    return {
        "active_parameter_count_mean": float(sum(active_counts) / len(active_counts)),
        "active_parameter_count_min": int(min(active_counts)),
        "active_parameter_count_max": int(max(active_counts)),
        "active_feature_union": sorted(feature_counter),
        "active_feature_frequency": feature_frequency,
        "leading_coefficients": leading_coefficients[:8],
        "coefficient_columns": coefficient_columns,
    }


def _default_tune_result(model_name: str, *, search_grid: list[dict[str, Any]]) -> dict:
    config = penalized_glm_config(model_name)
    best_choice = penalty_choice(
        model_name,
        c_value=float(config.default_c),
        l1_ratio=config.default_l1_ratio,
    )
    best_lambda = float(best_choice["lambda"])
    best_params = {"lambda": best_lambda, "c": float(best_choice["c"])}
    if best_choice["l1_ratio"] is not None:
        best_params["l1_ratio"] = float(best_choice["l1_ratio"])
    return {
        "model_name": model_name,
        "penalty_family": str(best_choice["penalty_family"]),
        "best_params": best_params,
        "best_lambda": best_lambda,
        "best_c": float(best_choice["c"]),
        "best_l1_ratio": None if best_choice["l1_ratio"] is None else float(best_choice["l1_ratio"]),
        "best_penalty": best_choice,
        "search_grid": [dict(row) for row in search_grid],
        "best_fit_summary": {},
        "best_active_parameter_count": None,
        "best_active_features": [],
        "best_active_coefficient_summary": [],
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
    if not c_values:
        c_values = [float(config.default_c)]
    if config.penalty == "elasticnet" and not l1_ratio_values:
        l1_ratio_values = [float(config.default_l1_ratio if config.default_l1_ratio is not None else 0.5)]

    if config.penalty == "elasticnet":
        param_grid = [
            penalty_choice(model_name, c_value=float(c), l1_ratio=float(l1_ratio))
            for c in c_values
            for l1_ratio in l1_ratio_values
        ]
    else:
        param_grid = [penalty_choice(model_name, c_value=float(c)) for c in c_values]

    default_result = _default_tune_result(model_name, search_grid=param_grid)

    train = df[df["home_win"].notna()].copy().sort_values("start_time_utc")
    if len(train) < max(80, min_train_size + 20):
        return default_result

    splits = time_series_splits(train, n_splits=n_splits, min_train_size=min_train_size)
    if not splits:
        return default_result

    rows = []
    fold_rows = []
    for params in param_grid:
        fold_scores = []
        fold_fit_summaries: list[dict[str, Any]] = []
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
            fit_summary = _fit_summary(model)
            fold_scores.append(met)
            fold_fit_summaries.append(fit_summary)
            fold_rows.append(
                {
                    **params,
                    "fold": int(fold),
                    "n_train": int(len(tr)),
                    "n_valid": int(len(va)),
                    "log_loss": float(met["log_loss"]),
                    "brier": float(met["brier"]),
                    "accuracy": float(met["accuracy"]),
                    "auc": float(met["auc"]),
                    "active_parameter_count": fit_summary.get("active_parameter_count"),
                    "active_features": list(fit_summary.get("active_features", [])),
                    "leading_features": _leading_features(fit_summary),
                }
            )
        if not fold_scores:
            continue
        aggregate = _aggregate_fold_fit_summaries(fold_fit_summaries)
        rows.append(
            {
                **params,
                "folds_used": int(len(fold_scores)),
                "log_loss": float(pd.DataFrame(fold_scores)["log_loss"].mean()),
                "brier": float(pd.DataFrame(fold_scores)["brier"].mean()),
                "accuracy": float(pd.DataFrame(fold_scores)["accuracy"].mean()),
                "auc": float(pd.DataFrame(fold_scores)["auc"].mean()),
                **aggregate,
            }
        )

    if not rows:
        out = dict(default_result)
        out["fold_metrics"] = fold_rows
        return out

    best = sorted(rows, key=lambda r: (r["log_loss"], r["brier"], -r["accuracy"]))[0]
    best_choice = penalty_choice(
        model_name,
        lambda_value=float(best["lambda"]),
        c_value=float(best["c"]),
        l1_ratio=best.get("l1_ratio"),
    )
    best_params = {"lambda": float(best_choice["lambda"]), "c": float(best_choice["c"])}
    best_l1_ratio = None if best_choice["l1_ratio"] is None else float(best_choice["l1_ratio"])
    if best_l1_ratio is not None:
        best_params["l1_ratio"] = best_l1_ratio
    best_fit_summary: dict[str, Any] = {}
    best_model = build_penalized_glm(
        model_name,
        c=float(best_choice["c"]),
        l1_ratio=best_choice.get("l1_ratio"),
    )
    best_model.fit(train, feature_cols)
    best_fit_summary = _fit_summary(best_model, top_n=10)

    return {
        "model_name": model_name,
        "penalty_family": str(best_choice["penalty_family"]),
        "best_params": best_params,
        "best_lambda": float(best_choice["lambda"]),
        "best_c": float(best_choice["c"]),
        "best_l1_ratio": best_l1_ratio,
        "best_penalty": best_choice,
        "search_grid": [dict(row) for row in param_grid],
        "best_fit_summary": best_fit_summary,
        "best_active_parameter_count": best_fit_summary.get("active_parameter_count"),
        "best_active_features": list(best_fit_summary.get("active_features", [])),
        "best_active_coefficient_summary": list(best_fit_summary.get("active_coefficient_summary", [])),
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
