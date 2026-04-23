from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, probplot

from src.common.utils import ensure_dir

DENSE_RESIDUAL_THRESHOLD = 250
MAX_BINNED_POINTS = 100
SCATTER_POINT_CAP = 4000

FEATURE_SUMMARY_COLUMNS = [
    "feature",
    "coef_scaled",
    "coef_original",
    "n_non_missing",
    "n_imputed",
    "n_unique_non_missing",
    "bin_count",
    "working_residual_plot_file",
    "partial_residual_plot_file",
]
LINEAR_PREDICTOR_BIN_COLUMNS = [
    "bin_index",
    "n_obs",
    "working_weight_sum",
    "linear_predictor_mean",
    "linear_predictor_min",
    "linear_predictor_max",
    "working_residual_mean",
]
FEATURE_WORKING_BIN_COLUMNS = [
    "feature",
    "bin_index",
    "n_obs",
    "working_weight_sum",
    "feature_value_mean",
    "feature_value_min",
    "feature_value_max",
    "working_residual_mean",
]
WEIGHT_BIN_COLUMNS = [
    "bin_index",
    "n_obs",
    "working_weight_sum",
    "weight_value_mean",
    "weight_value_min",
    "weight_value_max",
    "working_residual_mean",
]
PARTIAL_BIN_COLUMNS = [
    "feature",
    "bin_index",
    "n_obs",
    "working_weight_sum",
    "feature_value_mean",
    "feature_value_min",
    "feature_value_max",
    "partial_residual_mean",
    "component_mean",
]
RAW_RESIDUAL_COLUMNS = [
    "row_index",
    "fitted_probability",
    "linear_predictor",
    "working_residual",
    "deviance_residual",
    "randomized_quantile_residual",
]
RESIDUAL_VS_FITTED_COLUMNS = [
    "bin_index",
    "n_obs",
    "working_weight_sum",
    "fitted_probability_mean",
    "fitted_probability_min",
    "fitted_probability_max",
    "working_residual_mean",
    "deviance_residual_mean",
    "randomized_quantile_residual_mean",
]
GROUPED_RESIDUAL_COLUMNS = [
    "grouping_dimension",
    "group_value",
    "n_obs",
    "fitted_probability_mean",
    "working_residual_mean",
    "working_residual_mae",
    "deviance_residual_mean",
    "deviance_residual_mae",
    "randomized_quantile_residual_mean",
    "randomized_quantile_residual_mae",
]


def _numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _slugify(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(token)).strip("_")
    return cleaned or "feature"


def _safe_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return float(value)
    return value


def working_residual_logit(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    fitted = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    observed = np.asarray(y, dtype=float)
    return (observed - fitted) / (fitted * (1.0 - fitted))


def working_weight_logit(p: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    fitted = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = fitted * (1.0 - fitted)
    if sample_weight is None:
        return base
    return np.asarray(sample_weight, dtype=float) * base


def _deviance_residual_binary(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    observed = np.asarray(y, dtype=float)
    fitted = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    term = observed * np.log(np.clip(observed / fitted, 1e-6, None))
    term += (1.0 - observed) * np.log(np.clip((1.0 - observed) / (1.0 - fitted), 1e-6, None))
    return np.sign(observed - fitted) * np.sqrt(2.0 * term)


def randomized_quantile_residual_binary(
    y: np.ndarray,
    p: np.ndarray,
    *,
    random_state: int = 42,
) -> np.ndarray:
    observed = np.asarray(y, dtype=float)
    fitted = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lower = np.where(observed >= 1.0, 1.0 - fitted, 0.0)
    upper = np.where(observed >= 1.0, 1.0, 1.0 - fitted)
    rng = np.random.default_rng(random_state)
    u = rng.uniform(lower, upper)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def _bin_count(n_obs: int, n_unique: int) -> int:
    if n_obs <= 0 or n_unique <= 0:
        return 0
    target = int(np.sqrt(max(n_obs, 1)))
    target = max(4, target) if n_obs < DENSE_RESIDUAL_THRESHOLD else max(12, target)
    return int(min(MAX_BINNED_POINTS, n_obs, n_unique, max(2, target)))


def _scatter_indices(n_obs: int) -> np.ndarray:
    if n_obs <= SCATTER_POINT_CAP:
        return np.arange(n_obs, dtype=int)
    return np.linspace(0, n_obs - 1, SCATTER_POINT_CAP, dtype=int)


def _aggregate_equal_weight_bins(
    axis_values: np.ndarray,
    working_weights: np.ndarray,
    columns: dict[str, np.ndarray],
    *,
    n_bins: int,
) -> pd.DataFrame:
    if n_bins <= 0:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "axis_value": np.asarray(axis_values, dtype=float),
            "working_weight": np.asarray(working_weights, dtype=float),
        }
    )
    for key, values in columns.items():
        frame[key] = np.asarray(values, dtype=float)

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["axis_value", "working_weight"])
    if frame.empty:
        return pd.DataFrame()

    frame["working_weight"] = frame["working_weight"].clip(lower=1e-9)
    frame = frame.sort_values("axis_value", kind="mergesort").reset_index(drop=True)
    total_weight = float(frame["working_weight"].sum())
    if total_weight <= 0:
        return pd.DataFrame()

    cuts = np.linspace(total_weight / n_bins, total_weight, n_bins)
    frame["bin_index"] = np.searchsorted(cuts[:-1], frame["working_weight"].cumsum().to_numpy(dtype=float), side="right")

    rows: list[dict[str, Any]] = []
    for raw_bin_index, bucket in frame.groupby("bin_index", sort=True):
        weights = bucket["working_weight"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "bin_index": int(raw_bin_index + 1),
            "n_obs": int(len(bucket)),
            "working_weight_sum": float(weights.sum()),
            "axis_mean": float(np.average(bucket["axis_value"], weights=weights)),
            "axis_min": float(bucket["axis_value"].min()),
            "axis_max": float(bucket["axis_value"].max()),
        }
        for key in columns:
            row[f"{key}_mean"] = float(np.average(bucket[key].to_numpy(dtype=float), weights=weights))
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_working_residuals(
    axis_values: np.ndarray,
    residuals: np.ndarray,
    binned: pd.DataFrame,
    *,
    binned_column: str = "working_residual_mean",
    x_label: str,
    y_label: str = "Working residual",
    title: str,
    out_path: Path,
) -> None:
    sample = _scatter_indices(len(axis_values))
    plt.figure(figsize=(8, 4.5))
    plt.scatter(axis_values[sample], residuals[sample], s=10, alpha=0.18, color="#94a3b8", edgecolors="none")
    if not binned.empty:
        plt.plot(
            binned["axis_mean"],
            binned[binned_column],
            color="#0f766e",
            marker="o",
            markersize=4,
            linewidth=1.6,
        )
    plt.axhline(0.0, color="#111827", linewidth=1.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_histogram_with_normal(
    values: np.ndarray,
    *,
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return
    mu = float(np.mean(sample))
    sigma = float(np.std(sample, ddof=0))
    plt.figure(figsize=(8, 4.5))
    plt.hist(sample, bins=min(40, max(10, int(np.sqrt(sample.size)))), density=True, color="#cbd5e1", edgecolor="#94a3b8")
    if sigma > 0:
        xs = np.linspace(float(sample.min()), float(sample.max()), 200)
        plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), color="#0f172a", linewidth=1.6)
    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_qq(
    values: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return
    theoretical, ordered = probplot(sample, dist="norm", fit=False)
    slope, intercept = np.polyfit(np.asarray(theoretical, dtype=float), np.asarray(ordered, dtype=float), 1)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(theoretical, ordered, s=14, alpha=0.7, color="#1d4ed8", edgecolors="none")
    xs = np.linspace(float(np.min(theoretical)), float(np.max(theoretical)), 100)
    plt.plot(xs, intercept + slope * xs, color="#b45309", linewidth=1.5)
    plt.xlabel("Theoretical normal quantile")
    plt.ylabel("Sample quantile")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _weight_axis(df: pd.DataFrame) -> tuple[str, np.ndarray]:
    for col in ("sample_weight", "weight", "weights", "exposure", "exposures"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=float)
            return col, values
    return "unit_weight", np.ones(len(df), dtype=float)


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "mae": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": int(sample.size),
        "mean": _safe_float(float(np.mean(sample))),
        "std": _safe_float(float(np.std(sample, ddof=0))),
        "mae": _safe_float(float(np.mean(np.abs(sample)))),
        "p05": _safe_float(float(np.percentile(sample, 5.0))),
        "p50": _safe_float(float(np.percentile(sample, 50.0))),
        "p95": _safe_float(float(np.percentile(sample, 95.0))),
    }


def _resolve_feature_medians(glm: Any) -> pd.Series:
    for candidate in (getattr(glm, "feature_medians", None), getattr(getattr(glm, "_candidate", None), "medians", None)):
        if isinstance(candidate, dict):
            return pd.Series(candidate, dtype=float)
        if isinstance(candidate, pd.Series):
            return candidate.astype(float)
    return pd.Series(dtype=float)


def _resolve_scaler(glm: Any) -> Any | None:
    scaler = getattr(glm, "scaler", None)
    if scaler is not None and hasattr(scaler, "transform"):
        return scaler
    nested = getattr(getattr(glm, "_candidate", None), "scaler", None)
    if nested is not None and hasattr(nested, "transform"):
        return nested
    return None


def _resolve_linear_predictor(glm: Any, x_scaled: np.ndarray, fitted: np.ndarray) -> tuple[np.ndarray, str]:
    model = getattr(glm, "model", None)
    if model is not None and hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(x_scaled), dtype=float), "model_decision_function"
        except Exception:
            pass
    return np.log(np.clip(fitted, 1e-6, 1 - 1e-6) / np.clip(1.0 - fitted, 1e-6, 1 - 1e-6)), "logit_of_fitted_probability"


def _resolve_scaled_coefficients(glm: Any, feature_cols: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not feature_cols:
        return None, None

    model = getattr(glm, "model", None)
    scaler = _resolve_scaler(glm)
    if model is not None and hasattr(model, "coef_") and scaler is not None and hasattr(scaler, "scale_"):
        coef_scaled = np.asarray(model.coef_[0], dtype=float)
        if coef_scaled.size != len(feature_cols):
            return None, None
        scale = np.asarray(scaler.scale_, dtype=float)
        coef_original = np.divide(
            coef_scaled,
            scale,
            out=np.full_like(coef_scaled, np.nan, dtype=float),
            where=np.isfinite(scale) & (np.abs(scale) > 0),
        )
        return coef_scaled, coef_original

    params_payload = None
    exog_names: list[str] = []
    feature_scales: dict[str, float] = {}

    result = getattr(glm, "result", None)
    if result is not None:
        params_payload = getattr(result, "params", None)
        exog_names = list(getattr(glm, "exog_names", []) or [])
        if isinstance(getattr(glm, "feature_scales", None), dict):
            feature_scales = {str(key): float(value) for key, value in getattr(glm, "feature_scales", {}).items()}
    if params_payload is None:
        candidate = getattr(glm, "_candidate", None)
        result = getattr(candidate, "result", None) if candidate is not None else None
        if result is not None:
            params_payload = getattr(result, "params", None)
            exog_names = list(getattr(candidate, "exog_names", []) or [])

    if params_payload is None:
        return None, None

    params = np.asarray(params_payload, dtype=float)
    if params.size == 0:
        return None, None
    param_series = pd.Series(params, index=exog_names if len(exog_names) == len(params) else None)

    if scaler is not None and hasattr(scaler, "scale_"):
        scale_values = np.asarray(scaler.scale_, dtype=float)
        scale_by_feature = {
            feature: float(scale_values[idx])
            for idx, feature in enumerate(feature_cols)
            if idx < len(scale_values)
        }
    else:
        scale_by_feature = {}
    scale_by_feature |= feature_scales

    coef_scaled = []
    coef_original = []
    for feature in feature_cols:
        value = float(param_series.get(feature, np.nan))
        coef_scaled.append(value)
        scale = float(scale_by_feature.get(feature, 1.0))
        if np.isfinite(value) and np.isfinite(scale) and abs(scale) > 0:
            coef_original.append(float(value / scale))
        else:
            coef_original.append(np.nan)
    return np.asarray(coef_scaled, dtype=float), np.asarray(coef_original, dtype=float)


def _grouped_residual_summaries(
    df: pd.DataFrame,
    residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    candidates: dict[str, pd.Series] = {}
    for column in ("season", "home_team", "away_team", "bookmaker_key", "market_key"):
        if column in df.columns:
            candidates[column] = df[column]
    for date_column in ("game_date_utc", "start_time_utc"):
        if date_column not in df.columns:
            continue
        parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True)
        if parsed.notna().any():
            candidates["game_date"] = parsed.dt.date.astype(str)
            candidates["game_month"] = parsed.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
            break

    for dimension, values in candidates.items():
        frame = residuals.copy()
        frame["group_value"] = values.astype(str)
        frame = frame.replace({"group_value": {"": np.nan, "nan": np.nan, "NaT": np.nan}}).dropna(subset=["group_value"])
        if frame.empty:
            metadata_rows.append(
                {
                    "grouping_dimension": dimension,
                    "status": "not_applicable",
                    "reason": "no_non_null_group_values",
                }
            )
            continue
        n_groups = int(frame["group_value"].nunique())
        if n_groups < 2:
            metadata_rows.append(
                {
                    "grouping_dimension": dimension,
                    "status": "not_applicable",
                    "reason": "insufficient_group_cardinality",
                    "n_groups": n_groups,
                }
            )
            continue

        grouped = frame.groupby("group_value", dropna=True, sort=True)
        for group_value, bucket in grouped:
            summary_rows.append(
                {
                    "grouping_dimension": dimension,
                    "group_value": str(group_value),
                    "n_obs": int(len(bucket)),
                    "fitted_probability_mean": _safe_float(float(bucket["fitted_probability"].mean())),
                    "working_residual_mean": _safe_float(float(bucket["working_residual"].mean())),
                    "working_residual_mae": _safe_float(float(bucket["working_residual"].abs().mean())),
                    "deviance_residual_mean": _safe_float(float(bucket["deviance_residual"].mean())),
                    "deviance_residual_mae": _safe_float(float(bucket["deviance_residual"].abs().mean())),
                    "randomized_quantile_residual_mean": _safe_float(float(bucket["randomized_quantile_residual"].mean())),
                    "randomized_quantile_residual_mae": _safe_float(float(bucket["randomized_quantile_residual"].abs().mean())),
                }
            )
        metadata_rows.append(
            {
                "grouping_dimension": dimension,
                "status": "applicable",
                "n_groups": n_groups,
                "n_rows": int(len(frame)),
            }
        )

    if not metadata_rows:
        metadata_rows.append(
            {
                "grouping_dimension": "none",
                "status": "not_applicable",
                "reason": "no_supported_grouping_columns",
            }
        )

    grouped_summary = pd.DataFrame(summary_rows, columns=GROUPED_RESIDUAL_COLUMNS)
    metadata = {
        "grouping_dimensions": metadata_rows,
        "status": "ok" if not grouped_summary.empty else "partial",
        "supported_dimension_count": int(sum(1 for row in metadata_rows if row.get("status") == "applicable")),
    }
    return grouped_summary, metadata


def _plot_partial_residuals(
    axis_values: np.ndarray,
    partial_residuals: np.ndarray,
    component: np.ndarray,
    binned: pd.DataFrame,
    *,
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    sample = _scatter_indices(len(axis_values))
    order = np.argsort(axis_values, kind="mergesort")
    plt.figure(figsize=(8, 4.5))
    plt.scatter(axis_values[sample], partial_residuals[sample], s=10, alpha=0.18, color="#94a3b8", edgecolors="none")
    if not binned.empty:
        plt.plot(
            binned["axis_mean"],
            binned["partial_residual_mean"],
            color="#1d4ed8",
            marker="o",
            markersize=4,
            linewidth=1.6,
            label="Binned partial residual",
        )
        plt.plot(
            binned["axis_mean"],
            binned["component_mean"],
            color="#b45309",
            linewidth=1.5,
            linestyle="--",
            label="Model component",
        )
    else:
        plt.plot(axis_values[order], component[order], color="#b45309", linewidth=1.5, linestyle="--", label="Model component")
    plt.axhline(0.0, color="#111827", linewidth=1.0)
    plt.xlabel(x_label)
    plt.ylabel("Partial residual")
    plt.title(title)
    if not binned.empty:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_glm_diagnostics(
    df: pd.DataFrame,
    *,
    glm: Any,
    target_col: str,
    out_dir: str,
    prefix: str = "glm",
    relative_plot_dir: str = "plots",
) -> dict[str, Any]:
    plot_root = ensure_dir(Path(out_dir))
    plot_rel = str(relative_plot_dir).strip("/").replace("\\", "/") or "plots"
    feature_cols = list(getattr(glm, "feature_columns", []) or [])
    work = df[df[target_col].notna()].copy()

    empty = {
        "summary": {
            "status": "insufficient_data",
            "headline": "GLM residual diagnostics skipped because the fitted model or target data was unavailable",
            "n_observations": int(len(work)),
            "n_features": int(len(feature_cols)),
        },
        "feature_summary": pd.DataFrame(columns=FEATURE_SUMMARY_COLUMNS),
        "linear_predictor_bins": pd.DataFrame(columns=LINEAR_PREDICTOR_BIN_COLUMNS),
        "feature_working_bins": pd.DataFrame(columns=FEATURE_WORKING_BIN_COLUMNS),
        "weight_bins": pd.DataFrame(columns=WEIGHT_BIN_COLUMNS),
        "partial_residual_bins": pd.DataFrame(columns=PARTIAL_BIN_COLUMNS),
        "raw_residuals": pd.DataFrame(columns=RAW_RESIDUAL_COLUMNS),
        "residual_vs_fitted_bins": pd.DataFrame(columns=RESIDUAL_VS_FITTED_COLUMNS),
        "grouped_residual_summary": pd.DataFrame(columns=GROUPED_RESIDUAL_COLUMNS),
        "grouped_residual_metadata": {
            "status": "not_applicable",
            "grouping_dimensions": [],
            "supported_dimension_count": 0,
        },
    }
    if work.empty or glm is None or not feature_cols:
        return empty

    numeric = _numeric_frame(work, feature_cols)
    medians = _resolve_feature_medians(glm)
    filled = numeric.fillna(medians).fillna(0.0)
    y = work[target_col].astype(int).to_numpy(dtype=float)
    fitted = np.clip(np.asarray(glm.predict_proba(work), dtype=float), 1e-6, 1 - 1e-6)
    scaler = _resolve_scaler(glm)
    try:
        x_scaled = scaler.transform(filled.to_numpy(dtype=float)) if scaler is not None else filled.to_numpy(dtype=float)
    except Exception:
        x_scaled = filled.to_numpy(dtype=float)
    linear_predictor, linear_predictor_method = _resolve_linear_predictor(glm, x_scaled, fitted)
    working_residual = working_residual_logit(y, fitted)
    working_weight = working_weight_logit(fitted)
    deviance_residual = _deviance_residual_binary(y, fitted)
    randomized_quantile_residual = randomized_quantile_residual_binary(y, fitted)
    weight_axis_name, weight_axis_values = _weight_axis(work)

    deviance_path = plot_root / f"{prefix}_deviance_residuals.png"
    _plot_working_residuals(
        fitted,
        deviance_residual,
        pd.DataFrame(),
        x_label="Fitted probability",
        y_label="Deviance residual",
        title="Deviance residuals vs fitted probability",
        out_path=deviance_path,
    )
    deviance_hist_path = plot_root / f"{prefix}_deviance_residuals_histogram.png"
    _plot_histogram_with_normal(
        deviance_residual,
        x_label="Deviance residual",
        title="Deviance residual histogram",
        out_path=deviance_hist_path,
    )
    deviance_qq_path = plot_root / f"{prefix}_deviance_residuals_qq.png"
    _plot_qq(
        deviance_residual,
        title="Deviance residual normal Q-Q plot",
        out_path=deviance_qq_path,
    )
    randomized_hist_path = plot_root / f"{prefix}_randomized_quantile_residuals_histogram.png"
    _plot_histogram_with_normal(
        randomized_quantile_residual,
        x_label="Randomized quantile residual",
        title="Randomized quantile residual histogram",
        out_path=randomized_hist_path,
    )
    randomized_qq_path = plot_root / f"{prefix}_randomized_quantile_residuals_qq.png"
    _plot_qq(
        randomized_quantile_residual,
        title="Randomized quantile residual normal Q-Q plot",
        out_path=randomized_qq_path,
    )

    fitted_bins = _aggregate_equal_weight_bins(
        fitted,
        working_weight,
        {
            "working_residual": working_residual,
            "deviance_residual": deviance_residual,
            "randomized_quantile_residual": randomized_quantile_residual,
        },
        n_bins=_bin_count(len(work), len(np.unique(np.round(fitted, 12)))),
    )
    fitted_plot_path = plot_root / f"{prefix}_working_residuals_fitted_probability.png"
    _plot_working_residuals(
        fitted,
        working_residual,
        fitted_bins,
        x_label="Fitted probability",
        title="Working residuals vs fitted probability",
        out_path=fitted_plot_path,
    )

    linear_bins = _aggregate_equal_weight_bins(
        linear_predictor,
        working_weight,
        {"working_residual": working_residual},
        n_bins=_bin_count(len(work), len(np.unique(np.round(linear_predictor, 12)))),
    )
    linear_plot_path = plot_root / f"{prefix}_working_residuals_linear_predictor.png"
    _plot_working_residuals(
        linear_predictor,
        working_residual,
        linear_bins,
        x_label="Linear predictor",
        title="Working residuals vs linear predictor",
        out_path=linear_plot_path,
    )
    weight_unique_count = int(len(np.unique(np.round(weight_axis_values, 12))))
    weight_plot_path: Path | None = None
    if weight_unique_count >= 2:
        weight_bins = _aggregate_equal_weight_bins(
            weight_axis_values,
            working_weight,
            {"working_residual": working_residual},
            n_bins=_bin_count(len(work), weight_unique_count),
        )
        weight_plot_path = plot_root / f"{prefix}_working_residuals_weight.png"
        _plot_working_residuals(
            weight_axis_values,
            working_residual,
            weight_bins,
            x_label=weight_axis_name,
            title=f"Working residuals vs {weight_axis_name}",
            out_path=weight_plot_path,
        )
    else:
        weight_bins = pd.DataFrame(columns=["bin_index", "n_obs", "working_weight_sum", "axis_mean", "axis_min", "axis_max", "working_residual_mean"])

    coef_scaled, coef_original = _resolve_scaled_coefficients(glm, feature_cols)
    has_partial_residual_support = (
        coef_scaled is not None
        and coef_original is not None
        and len(coef_scaled) == len(feature_cols)
        and len(coef_original) == len(feature_cols)
        and x_scaled.shape[1] == len(feature_cols)
    )

    feature_rows: list[dict[str, Any]] = []
    working_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(feature_cols):
        values = filled[feature].to_numpy(dtype=float)
        non_missing = numeric[feature].dropna()
        component = x_scaled[:, idx] * coef_scaled[idx] if has_partial_residual_support else np.full(len(values), np.nan)
        partial_residual = working_residual + component if has_partial_residual_support else np.full(len(values), np.nan)
        n_bins = _bin_count(len(work), int(non_missing.nunique()) if not non_missing.empty else 0)

        columns = {"working_residual": working_residual}
        if has_partial_residual_support:
            columns["partial_residual"] = partial_residual
            columns["component"] = component
        binned = _aggregate_equal_weight_bins(
            values,
            working_weight,
            columns,
            n_bins=n_bins,
        )

        working_plot_path = plot_root / f"{prefix}_working_residuals_{_slugify(feature)}.png"
        partial_plot_path = plot_root / f"{prefix}_partial_residuals_{_slugify(feature)}.png"
        _plot_working_residuals(
            values,
            working_residual,
            binned[["axis_mean", "working_residual_mean"]] if not binned.empty else pd.DataFrame(),
            x_label=feature,
            title=f"Working residuals vs {feature}",
            out_path=working_plot_path,
        )
        if has_partial_residual_support:
            _plot_partial_residuals(
                values,
                partial_residual,
                component,
                binned[["axis_mean", "partial_residual_mean", "component_mean"]] if not binned.empty else pd.DataFrame(),
                x_label=feature,
                title=f"Partial residuals vs {feature}",
                out_path=partial_plot_path,
            )

        feature_rows.append(
            {
                "feature": feature,
                "coef_scaled": _safe_float(float(coef_scaled[idx])) if has_partial_residual_support else None,
                "coef_original": _safe_float(coef_original[idx]) if has_partial_residual_support else None,
                "n_non_missing": int(non_missing.shape[0]),
                "n_imputed": int(numeric[feature].isna().sum()),
                "n_unique_non_missing": int(non_missing.nunique()) if not non_missing.empty else 0,
                "bin_count": int(len(binned)),
                "working_residual_plot_file": f"{plot_rel}/{working_plot_path.name}",
                "partial_residual_plot_file": f"{plot_rel}/{partial_plot_path.name}" if has_partial_residual_support else None,
            }
        )

        for _, row in binned.iterrows():
            base = {
                "feature": feature,
                "bin_index": int(row["bin_index"]),
                "n_obs": int(row["n_obs"]),
                "working_weight_sum": float(row["working_weight_sum"]),
                "feature_value_mean": float(row["axis_mean"]),
                "feature_value_min": float(row["axis_min"]),
                "feature_value_max": float(row["axis_max"]),
            }
            working_rows.append(base | {"working_residual_mean": float(row["working_residual_mean"])})
            if has_partial_residual_support:
                partial_rows.append(
                    base
                    | {
                        "partial_residual_mean": float(row["partial_residual_mean"]),
                        "component_mean": float(row["component_mean"]),
                    }
                )

    linear_predictor_bins = linear_bins.rename(
        columns={
            "axis_mean": "linear_predictor_mean",
            "axis_min": "linear_predictor_min",
            "axis_max": "linear_predictor_max",
        }
    )
    residual_vs_fitted_bins = fitted_bins.rename(
        columns={
            "axis_mean": "fitted_probability_mean",
            "axis_min": "fitted_probability_min",
            "axis_max": "fitted_probability_max",
        }
    )
    weight_bins = weight_bins.rename(
        columns={
            "axis_mean": "weight_value_mean",
            "axis_min": "weight_value_min",
            "axis_max": "weight_value_max",
        }
    )

    raw_residuals = pd.DataFrame(
        {
            "row_index": np.arange(len(work), dtype=int),
            "fitted_probability": fitted,
            "linear_predictor": linear_predictor,
            "working_residual": working_residual,
            "deviance_residual": deviance_residual,
            "randomized_quantile_residual": randomized_quantile_residual,
        },
        columns=RAW_RESIDUAL_COLUMNS,
    )
    grouped_residual_summary, grouped_residual_metadata = _grouped_residual_summaries(work, raw_residuals)

    model_label = str(getattr(glm, "model_name", "glm")).strip() or "glm"
    summary = {
        "status": "ok",
        "headline": f"Generated working residual, binned working residual, and partial residual diagnostics for {model_label}",
        "sample_name": "train_df",
        "n_observations": int(len(work)),
        "n_features": int(len(feature_cols)),
        "working_weight_sum": _safe_float(float(working_weight.sum())),
        "variance_rule_recommended_max_bins": int(np.floor(0.01 * float(working_weight.sum()))),
        "linear_predictor_plot_file": f"{plot_rel}/{linear_plot_path.name}",
        "working_vs_fitted_plot_file": f"{plot_rel}/{fitted_plot_path.name}",
        "linear_predictor_method": linear_predictor_method,
        "deviance_plot_file": f"{plot_rel}/{deviance_path.name}",
        "deviance_histogram_plot_file": f"{plot_rel}/{deviance_hist_path.name}",
        "deviance_qq_plot_file": f"{plot_rel}/{deviance_qq_path.name}",
        "randomized_quantile_histogram_plot_file": f"{plot_rel}/{randomized_hist_path.name}",
        "randomized_quantile_qq_plot_file": f"{plot_rel}/{randomized_qq_path.name}",
        "weight_plot_file": f"{plot_rel}/{weight_plot_path.name}" if weight_plot_path is not None else None,
        "weight_plot_status": "ok" if weight_plot_path is not None else "skipped_constant_axis",
        "weight_axis_name": weight_axis_name,
        "working_residual_definition": "wri = (y - m) / (m * (1 - m))",
        "partial_residual_definition": "partial = wri + beta_j * z_j",
        "binning_note": "Binned residual means use equal-working-weight bins and working-weighted averages.",
        "component_note": (
            f"beta_j * z_j uses fitted standardized design-matrix columns for {model_label}."
            if has_partial_residual_support
            else f"Partial residual decomposition is not available for {model_label} with the current wrapper payload."
        ),
        "partial_residual_status": "ok" if has_partial_residual_support else "partial",
        "raw_residual_summary": {
            "working_residual": _distribution_summary(working_residual),
            "deviance_residual": _distribution_summary(deviance_residual),
            "randomized_quantile_residual": _distribution_summary(randomized_quantile_residual),
        },
        "grouped_residual_metadata": grouped_residual_metadata,
        "deviance_residual_mean": _safe_float(float(np.mean(deviance_residual))),
        "deviance_residual_std": _safe_float(float(np.std(deviance_residual, ddof=0))),
        "randomized_quantile_residual_mean": _safe_float(float(np.mean(randomized_quantile_residual))),
        "randomized_quantile_residual_std": _safe_float(float(np.std(randomized_quantile_residual, ddof=0))),
    }

    return {
        "summary": summary,
        "feature_summary": pd.DataFrame(feature_rows, columns=FEATURE_SUMMARY_COLUMNS),
        "linear_predictor_bins": linear_predictor_bins.reindex(columns=LINEAR_PREDICTOR_BIN_COLUMNS),
        "feature_working_bins": pd.DataFrame(working_rows, columns=FEATURE_WORKING_BIN_COLUMNS),
        "weight_bins": weight_bins.reindex(columns=WEIGHT_BIN_COLUMNS),
        "partial_residual_bins": pd.DataFrame(partial_rows, columns=PARTIAL_BIN_COLUMNS),
        "raw_residuals": raw_residuals.reindex(columns=RAW_RESIDUAL_COLUMNS),
        "residual_vs_fitted_bins": residual_vs_fitted_bins.reindex(columns=RESIDUAL_VS_FITTED_COLUMNS),
        "grouped_residual_summary": grouped_residual_summary.reindex(columns=GROUPED_RESIDUAL_COLUMNS),
        "grouped_residual_metadata": grouped_residual_metadata,
    }
