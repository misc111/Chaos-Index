from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.common.utils import ensure_dir
from src.evaluation.metrics import brier_score


def _clean_binary_probability_inputs(
    y_true: np.ndarray | pd.Series | list[float],
    p_pred: np.ndarray | pd.Series | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], 1e-6, 1 - 1e-6)
    return y, p


def _thin_curve_points(df: pd.DataFrame, max_points: int = 250) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    return df.iloc[np.unique(idx)].reset_index(drop=True)


def _safe_float(value: float | int | np.floating | np.integer) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _positive_share(y: np.ndarray) -> float:
    return _safe_float(float(np.mean(y))) if len(y) else float("nan")


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    margin = z * np.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * n)) / n) / denom
    return _safe_float(center - margin), _safe_float(center + margin)


def quantile_plot_report(y_true: np.ndarray, p_pred: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    y, p = _clean_binary_probability_inputs(y_true, p_pred)
    base_cols = [
        "quantile",
        "n_obs",
        "share_obs",
        "min_pred",
        "avg_pred",
        "max_pred",
        "actual_rate",
        "gap",
        "abs_gap",
    ]
    if len(y) == 0:
        return {
            "summary": {
                "n_obs": 0,
                "positive_rate": float("nan"),
                "bins_requested": int(bins),
                "bins_realized": 0,
                "mean_abs_calibration_gap": float("nan"),
                "max_abs_calibration_gap": float("nan"),
                "rmse_calibration_gap": float("nan"),
                "monotonicity_violations": 0,
                "first_bin_actual_rate": float("nan"),
                "last_bin_actual_rate": float("nan"),
                "actual_rate_lift_last_vs_first": float("nan"),
                "first_bin_avg_pred": float("nan"),
                "last_bin_avg_pred": float("nan"),
            },
            "curve": pd.DataFrame(columns=base_cols),
        }

    work = pd.DataFrame({"y": y, "p": p})
    q = max(1, min(int(bins), int(work["p"].nunique())))
    work["quantile"] = pd.qcut(work["p"], q=q, labels=False, duplicates="drop") + 1
    curve = (
        work.groupby("quantile", as_index=False)
        .agg(
            n_obs=("y", "count"),
            min_pred=("p", "min"),
            avg_pred=("p", "mean"),
            max_pred=("p", "max"),
            actual_rate=("y", "mean"),
        )
        .sort_values("quantile")
        .reset_index(drop=True)
    )
    curve["share_obs"] = curve["n_obs"] / max(len(work), 1)
    curve["gap"] = curve["avg_pred"] - curve["actual_rate"]
    curve["abs_gap"] = curve["gap"].abs()

    actual_rate = curve["actual_rate"].to_numpy(dtype=float)
    monotonicity_violations = int(np.sum(np.diff(actual_rate) < -1e-9)) if len(actual_rate) > 1 else 0
    summary = {
        "n_obs": int(len(work)),
        "positive_rate": _positive_share(y),
        "bins_requested": int(bins),
        "bins_realized": int(len(curve)),
        "mean_abs_calibration_gap": _safe_float(curve["abs_gap"].mean()),
        "max_abs_calibration_gap": _safe_float(curve["abs_gap"].max()),
        "rmse_calibration_gap": _safe_float(np.sqrt(np.mean(np.square(curve["gap"])))),
        "monotonicity_violations": monotonicity_violations,
        "first_bin_actual_rate": _safe_float(curve["actual_rate"].iloc[0]),
        "last_bin_actual_rate": _safe_float(curve["actual_rate"].iloc[-1]),
        "actual_rate_lift_last_vs_first": _safe_float(curve["actual_rate"].iloc[-1] - curve["actual_rate"].iloc[0]),
        "first_bin_avg_pred": _safe_float(curve["avg_pred"].iloc[0]),
        "last_bin_avg_pred": _safe_float(curve["avg_pred"].iloc[-1]),
    }
    return {"summary": summary, "curve": curve[base_cols]}


def lorenz_gini_report(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, Any]:
    y, p = _clean_binary_probability_inputs(y_true, p_pred)
    base_cols = [
        "point_index",
        "cumulative_share_obs",
        "cumulative_share_events",
        "line_of_equality",
        "perfect_curve",
    ]
    if len(y) == 0:
        return {
            "summary": {
                "n_obs": 0,
                "positive_count": 0,
                "positive_rate": float("nan"),
                "curve_area": float("nan"),
                "raw_gini": float("nan"),
                "perfect_raw_gini": float("nan"),
                "normalized_gini": float("nan"),
                "top_decile_event_capture": float("nan"),
                "top_quintile_event_capture": float("nan"),
            },
            "curve": pd.DataFrame(columns=base_cols),
        }

    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    n = len(y_sorted)
    positives = int(y_sorted.sum())
    x = np.arange(0, n + 1, dtype=float) / max(n, 1)
    if positives > 0:
        cumulative_events = np.concatenate([[0.0], np.cumsum(y_sorted) / positives])
        perfect_curve = np.concatenate([[0.0], np.minimum(np.arange(1, n + 1), positives) / positives])
    else:
        cumulative_events = np.zeros(n + 1, dtype=float)
        perfect_curve = np.zeros(n + 1, dtype=float)

    curve = pd.DataFrame(
        {
            "point_index": np.arange(0, n + 1, dtype=int),
            "cumulative_share_obs": x,
            "cumulative_share_events": cumulative_events,
            "line_of_equality": x,
            "perfect_curve": perfect_curve,
        }
    )

    curve_area = _safe_float(np.trapz(cumulative_events, x))
    raw_gini = _safe_float(2.0 * curve_area - 1.0)
    perfect_raw_gini = _safe_float(2.0 * np.trapz(perfect_curve, x) - 1.0)
    normalized_gini = _safe_float(raw_gini / perfect_raw_gini) if perfect_raw_gini > 0 else float("nan")
    top_decile_n = max(1, int(np.ceil(0.10 * n)))
    top_quintile_n = max(1, int(np.ceil(0.20 * n)))
    top_decile_capture = _safe_float(y_sorted[:top_decile_n].sum() / positives) if positives > 0 else float("nan")
    top_quintile_capture = _safe_float(y_sorted[:top_quintile_n].sum() / positives) if positives > 0 else float("nan")

    summary = {
        "n_obs": int(n),
        "positive_count": positives,
        "positive_rate": _positive_share(y),
        "curve_area": curve_area,
        "raw_gini": raw_gini,
        "perfect_raw_gini": perfect_raw_gini,
        "normalized_gini": normalized_gini,
        "top_decile_event_capture": top_decile_capture,
        "top_quintile_event_capture": top_quintile_capture,
    }
    return {"summary": summary, "curve": _thin_curve_points(curve[base_cols])}


def _operating_point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    pos = tp + fn
    neg = tn + fp
    total = len(y)

    sensitivity = tp / pos if pos else float("nan")
    specificity = tn / neg if neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    accuracy = (tp + tn) / total if total else float("nan")
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0 if np.isfinite(sensitivity) and np.isfinite(specificity) else float("nan")
    )

    return {
        "threshold": _safe_float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": _safe_float(sensitivity),
        "specificity": _safe_float(specificity),
        "false_positive_rate": _safe_float(1.0 - specificity) if np.isfinite(specificity) else float("nan"),
        "precision": _safe_float(precision),
        "negative_predictive_value": _safe_float(npv),
        "accuracy": _safe_float(accuracy),
        "balanced_accuracy": _safe_float(balanced_accuracy),
        "predicted_positive_rate": _safe_float((tp + fp) / total if total else float("nan")),
    }


def _tossup_band_row(y: np.ndarray, p: np.ndarray, half_width: float) -> dict[str, float]:
    if half_width <= 0:
        tossup_mask = np.zeros(len(y), dtype=bool)
        decisive_mask = np.ones(len(y), dtype=bool)
    else:
        lower = 0.5 - half_width
        upper = 0.5 + half_width
        tossup_mask = (p >= lower) & (p <= upper)
        decisive_mask = ~tossup_mask

    lower = 0.5 - max(half_width, 0.0)
    upper = 0.5 + max(half_width, 0.0)
    decisive_y = y[decisive_mask]
    decisive_p = p[decisive_mask]
    tossup_y = y[tossup_mask]
    tossup_p = p[tossup_mask]

    decisive_accuracy = (
        _safe_float(np.mean((decisive_p >= 0.5).astype(int) == decisive_y)) if len(decisive_y) else float("nan")
    )
    decisive_brier = brier_score(decisive_y, decisive_p) if len(decisive_y) else float("nan")
    decisive_log_loss = (
        _safe_float(-np.mean(decisive_y * np.log(decisive_p) + (1 - decisive_y) * np.log(1 - decisive_p)))
        if len(decisive_y)
        else float("nan")
    )
    tossup_actual_rate = _positive_share(tossup_y) if len(tossup_y) else float("nan")
    tossup_mean_pred = _safe_float(np.mean(tossup_p)) if len(tossup_p) else float("nan")
    ci_low, ci_high = _wilson_interval(int(tossup_y.sum()), int(len(tossup_y)))

    return {
        "tossup_half_width": _safe_float(max(half_width, 0.0)),
        "lower_threshold": _safe_float(lower),
        "upper_threshold": _safe_float(upper),
        "decisive_count": int(decisive_mask.sum()),
        "tossup_count": int(tossup_mask.sum()),
        "decisive_share": _safe_float(decisive_mask.mean()) if len(y) else float("nan"),
        "tossup_share": _safe_float(tossup_mask.mean()) if len(y) else float("nan"),
        "decisive_accuracy": decisive_accuracy,
        "decisive_brier": _safe_float(decisive_brier),
        "decisive_log_loss": decisive_log_loss,
        "tossup_actual_rate": tossup_actual_rate,
        "tossup_mean_pred": tossup_mean_pred,
        "tossup_abs_gap_to_0_50": _safe_float(abs(tossup_actual_rate - 0.5)) if np.isfinite(tossup_actual_rate) else float("nan"),
        "tossup_actual_rate_ci_low": ci_low,
        "tossup_actual_rate_ci_high": ci_high,
        "tossup_ci_contains_0_50": bool(ci_low <= 0.5 <= ci_high) if np.isfinite(ci_low) and np.isfinite(ci_high) else False,
    }


def roc_report(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    tossup_half_widths: Sequence[float] | None = None,
    current_tossup_half_width: float = 0.05,
) -> dict[str, Any]:
    y, p = _clean_binary_probability_inputs(y_true, p_pred)
    curve_cols = ["false_positive_rate", "true_positive_rate", "threshold"]
    op_cols = [
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "sensitivity",
        "specificity",
        "false_positive_rate",
        "precision",
        "negative_predictive_value",
        "accuracy",
        "balanced_accuracy",
        "predicted_positive_rate",
    ]
    tossup_cols = [
        "tossup_half_width",
        "lower_threshold",
        "upper_threshold",
        "decisive_count",
        "tossup_count",
        "decisive_share",
        "tossup_share",
        "decisive_accuracy",
        "decisive_brier",
        "decisive_log_loss",
        "tossup_actual_rate",
        "tossup_mean_pred",
        "tossup_abs_gap_to_0_50",
        "tossup_actual_rate_ci_low",
        "tossup_actual_rate_ci_high",
        "tossup_ci_contains_0_50",
        "decisive_accuracy_gain_vs_no_abstain",
        "decisive_brier_gain_vs_no_abstain",
        "decisive_log_loss_gain_vs_no_abstain",
    ]
    if len(y) == 0:
        empty_curve = pd.DataFrame(columns=curve_cols)
        empty_ops = pd.DataFrame(columns=op_cols)
        empty_tossup = pd.DataFrame(columns=tossup_cols)
        summary = {
            "n_obs": 0,
            "positive_count": 0,
            "positive_rate": float("nan"),
            "auroc": float("nan"),
            "normalized_gini": float("nan"),
            "threshold_0_50_accuracy": float("nan"),
            "threshold_0_50_sensitivity": float("nan"),
            "threshold_0_50_specificity": float("nan"),
            "threshold_0_50_false_positive_rate": float("nan"),
            "threshold_0_50_predicted_positive_rate": float("nan"),
            "best_youden_threshold": float("nan"),
            "best_youden_j_stat": float("nan"),
            "best_youden_accuracy": float("nan"),
        }
        tossup_summary = {
            "tossup_lower_threshold": float("nan"),
            "tossup_upper_threshold": float("nan"),
            "tossup_half_width": float(current_tossup_half_width),
            "tossup_count": 0,
            "tossup_share": float("nan"),
            "tossup_actual_rate": float("nan"),
            "tossup_actual_rate_ci_low": float("nan"),
            "tossup_actual_rate_ci_high": float("nan"),
            "tossup_abs_gap_to_0_50": float("nan"),
            "decisive_count": 0,
            "decisive_share": float("nan"),
            "decisive_accuracy": float("nan"),
            "decisive_brier": float("nan"),
            "decisive_log_loss": float("nan"),
            "decisive_accuracy_gain_vs_no_abstain": float("nan"),
            "decisive_brier_gain_vs_no_abstain": float("nan"),
            "decisive_log_loss_gain_vs_no_abstain": float("nan"),
            "tossup_ci_contains_0_50": False,
        }
        return {
            "summary": summary,
            "curve": empty_curve,
            "operating_points": empty_ops,
            "tossup_sweep": empty_tossup,
            "tossup_summary": tossup_summary,
        }

    if tossup_half_widths is None:
        tossup_half_widths = [round(x, 2) for x in np.arange(0.00, 0.151, 0.01)]

    op_thresholds = sorted({0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95})
    operating_points = pd.DataFrame([_operating_point(y, p, threshold) for threshold in op_thresholds], columns=op_cols)
    threshold_050 = operating_points.loc[np.isclose(operating_points["threshold"], 0.5)].iloc[0].to_dict()

    if len(np.unique(y)) < 2:
        roc_curve_df = pd.DataFrame(
            {
                "false_positive_rate": [0.0, 1.0],
                "true_positive_rate": [0.0, 1.0],
                "threshold": [float("inf"), 0.0],
            }
        )
        auroc = float("nan")
        best_threshold = float("nan")
        best_j = float("nan")
        best_accuracy = float("nan")
    else:
        fpr, tpr, thresholds = roc_curve(y, p)
        roc_curve_df = pd.DataFrame(
            {
                "false_positive_rate": fpr,
                "true_positive_rate": tpr,
                "threshold": thresholds,
            }
        )
        auroc = _safe_float(roc_auc_score(y, p))
        valid = np.isfinite(thresholds)
        youden = tpr - fpr
        if np.any(valid):
            candidate_idx = np.where(valid)[0]
            best_idx = candidate_idx[int(np.nanargmax(youden[valid]))]
            best_threshold = _safe_float(thresholds[best_idx])
            best_j = _safe_float(youden[best_idx])
            best_accuracy = _operating_point(y, p, best_threshold)["accuracy"]
        else:
            best_threshold = float("nan")
            best_j = float("nan")
            best_accuracy = float("nan")

    tossup_rows = [_tossup_band_row(y, p, width) for width in tossup_half_widths]
    baseline_row = next((row for row in tossup_rows if abs(float(row["tossup_half_width"])) < 1e-12), tossup_rows[0])
    for row in tossup_rows:
        row["decisive_accuracy_gain_vs_no_abstain"] = (
            _safe_float(row["decisive_accuracy"] - baseline_row["decisive_accuracy"])
            if np.isfinite(row["decisive_accuracy"]) and np.isfinite(baseline_row["decisive_accuracy"])
            else float("nan")
        )
        row["decisive_brier_gain_vs_no_abstain"] = (
            _safe_float(baseline_row["decisive_brier"] - row["decisive_brier"])
            if np.isfinite(row["decisive_brier"]) and np.isfinite(baseline_row["decisive_brier"])
            else float("nan")
        )
        row["decisive_log_loss_gain_vs_no_abstain"] = (
            _safe_float(baseline_row["decisive_log_loss"] - row["decisive_log_loss"])
            if np.isfinite(row["decisive_log_loss"]) and np.isfinite(baseline_row["decisive_log_loss"])
            else float("nan")
        )
    tossup_sweep = pd.DataFrame(tossup_rows, columns=tossup_cols)
    current_row = tossup_sweep.loc[np.isclose(tossup_sweep["tossup_half_width"], current_tossup_half_width)]
    if current_row.empty:
        current_row = tossup_sweep.iloc[[int(np.argmin(np.abs(tossup_sweep["tossup_half_width"] - current_tossup_half_width)))]]
    current = current_row.iloc[0]

    summary = {
        "n_obs": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": _positive_share(y),
        "auroc": auroc,
        "normalized_gini": _safe_float(2.0 * auroc - 1.0) if np.isfinite(auroc) else float("nan"),
        "threshold_0_50_accuracy": _safe_float(threshold_050["accuracy"]),
        "threshold_0_50_sensitivity": _safe_float(threshold_050["sensitivity"]),
        "threshold_0_50_specificity": _safe_float(threshold_050["specificity"]),
        "threshold_0_50_false_positive_rate": _safe_float(threshold_050["false_positive_rate"]),
        "threshold_0_50_predicted_positive_rate": _safe_float(threshold_050["predicted_positive_rate"]),
        "best_youden_threshold": best_threshold,
        "best_youden_j_stat": best_j,
        "best_youden_accuracy": _safe_float(best_accuracy),
    }
    tossup_summary = {
        "tossup_lower_threshold": _safe_float(current["lower_threshold"]),
        "tossup_upper_threshold": _safe_float(current["upper_threshold"]),
        "tossup_half_width": _safe_float(current["tossup_half_width"]),
        "tossup_count": int(current["tossup_count"]),
        "tossup_share": _safe_float(current["tossup_share"]),
        "tossup_actual_rate": _safe_float(current["tossup_actual_rate"]),
        "tossup_actual_rate_ci_low": _safe_float(current["tossup_actual_rate_ci_low"]),
        "tossup_actual_rate_ci_high": _safe_float(current["tossup_actual_rate_ci_high"]),
        "tossup_abs_gap_to_0_50": _safe_float(current["tossup_abs_gap_to_0_50"]),
        "decisive_count": int(current["decisive_count"]),
        "decisive_share": _safe_float(current["decisive_share"]),
        "decisive_accuracy": _safe_float(current["decisive_accuracy"]),
        "decisive_brier": _safe_float(current["decisive_brier"]),
        "decisive_log_loss": _safe_float(current["decisive_log_loss"]),
        "decisive_accuracy_gain_vs_no_abstain": _safe_float(current["decisive_accuracy_gain_vs_no_abstain"]),
        "decisive_brier_gain_vs_no_abstain": _safe_float(current["decisive_brier_gain_vs_no_abstain"]),
        "decisive_log_loss_gain_vs_no_abstain": _safe_float(current["decisive_log_loss_gain_vs_no_abstain"]),
        "tossup_ci_contains_0_50": bool(current["tossup_ci_contains_0_50"]),
    }
    return {
        "summary": summary,
        "curve": _thin_curve_points(roc_curve_df[curve_cols]),
        "operating_points": operating_points[op_cols],
        "tossup_sweep": tossup_sweep[tossup_cols],
        "tossup_summary": tossup_summary,
    }


def save_probability_validation_plots(
    *,
    quantile_curve: pd.DataFrame,
    lorenz_curve: pd.DataFrame,
    roc_curve_df: pd.DataFrame,
    out_dir: str | Path,
    prefix: str = "glm",
) -> dict[str, str]:
    out = ensure_dir(Path(out_dir))
    paths: dict[str, str] = {}

    if not quantile_curve.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(quantile_curve["quantile"], quantile_curve["avg_pred"], marker="o", label="Average predicted probability")
        plt.plot(quantile_curve["quantile"], quantile_curve["actual_rate"], marker="o", label="Actual event rate")
        plt.xlabel("Probability quantile")
        plt.ylabel("Rate")
        plt.title("Logistic Holdout Quantile Plot")
        plt.ylim(0, 1)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        quantile_path = out / f"{prefix}_quantile_plot.png"
        plt.savefig(quantile_path, dpi=140)
        plt.close()
        paths["quantile_plot"] = str(quantile_path)

    if not lorenz_curve.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(lorenz_curve["cumulative_share_obs"], lorenz_curve["cumulative_share_events"], label="Model Lorenz curve")
        plt.plot(lorenz_curve["cumulative_share_obs"], lorenz_curve["line_of_equality"], linestyle="--", label="Line of equality")
        plt.plot(lorenz_curve["cumulative_share_obs"], lorenz_curve["perfect_curve"], linestyle=":", label="Perfect ordering")
        plt.xlabel("Cumulative share of games")
        plt.ylabel("Cumulative share of home-win events")
        plt.title("Logistic Lorenz Curve")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        lorenz_path = out / f"{prefix}_lorenz_curve.png"
        plt.savefig(lorenz_path, dpi=140)
        plt.close()
        paths["lorenz_curve"] = str(lorenz_path)

    if not roc_curve_df.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(roc_curve_df["false_positive_rate"], roc_curve_df["true_positive_rate"], label="ROC curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Line of equality")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Logistic ROC Curve")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        roc_path = out / f"{prefix}_roc_curve.png"
        plt.savefig(roc_path, dpi=140)
        plt.close()
        paths["roc_curve"] = str(roc_path)

    return paths


def validate_logistic_probability_model(
    y_true: np.ndarray | pd.Series | list[float],
    p_pred: np.ndarray | pd.Series | list[float],
    *,
    bins: int = 10,
    tossup_half_widths: Sequence[float] | None = None,
    current_tossup_half_width: float = 0.05,
    plot_dir: str | Path | None = None,
    plot_prefix: str = "glm",
) -> dict[str, Any]:
    quantile = quantile_plot_report(y_true, p_pred, bins=bins)
    lorenz = lorenz_gini_report(y_true, p_pred)
    roc = roc_report(
        y_true,
        p_pred,
        tossup_half_widths=tossup_half_widths,
        current_tossup_half_width=current_tossup_half_width,
    )
    plot_paths: dict[str, str] = {}
    if plot_dir is not None:
        plot_paths = save_probability_validation_plots(
            quantile_curve=quantile["curve"],
            lorenz_curve=lorenz["curve"],
            roc_curve_df=roc["curve"],
            out_dir=plot_dir,
            prefix=plot_prefix,
        )

    return {
        "quantile_summary": quantile["summary"],
        "quantile_curve": quantile["curve"],
        "lorenz_summary": lorenz["summary"],
        "lorenz_curve": lorenz["curve"],
        "roc_summary": roc["summary"],
        "roc_curve": roc["curve"],
        "operating_points": roc["operating_points"],
        "tossup_summary": roc["tossup_summary"],
        "tossup_sweep": roc["tossup_sweep"],
        "plot_paths": plot_paths,
    }
