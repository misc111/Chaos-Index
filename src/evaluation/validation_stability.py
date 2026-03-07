from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import metric_bundle
from src.models.glm_logit import GLMLogitModel
from src.training.cv import time_series_splits

PAIRWISE_WARN_THRESHOLD = 0.80
PAIRWISE_SEVERE_THRESHOLD = 0.95
VIF_WARN_THRESHOLD = 5.0
VIF_SEVERE_THRESHOLD = 10.0
CONDITION_WARN_THRESHOLD = 10.0
CONDITION_SEVERE_THRESHOLD = 30.0
VARIANCE_PROP_THRESHOLD = 0.50
DOMINANT_SHARE_THRESHOLD = 0.995
RELATIVE_NEAR_CONSTANT_STD_THRESHOLD = 1e-4
ABSOLUTE_NEAR_CONSTANT_STD_THRESHOLD = 1e-8
ZERO_STD_THRESHOLD = 1e-12
MIN_NON_MISSING = 3


def coefficient_paths(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "home_win",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [30, 60, 90]

    work = df[df[target_col].notna()].copy().sort_values("game_date_utc")
    if work.empty:
        return pd.DataFrame()

    out = []
    for w in windows:
        for i in range(w, len(work) + 1):
            seg = work.iloc[i - w : i]
            y = seg[target_col].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            x_frame = _safe_numeric_frame(seg, features)
            x = x_frame.fillna(x_frame.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
            m = LogisticRegression(max_iter=1500, C=1.0)
            m.fit(x, y)
            for f, c in zip(features, m.coef_[0]):
                out.append(
                    {
                        "window": w,
                        "as_of": seg.iloc[-1]["game_date_utc"],
                        "feature": f,
                        "coef": float(c),
                    }
                )
    return pd.DataFrame(out)


def _safe_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _dominant_share(series: pd.Series) -> float:
    non_missing = series.dropna()
    if non_missing.empty:
        return float("nan")
    counts = non_missing.value_counts(normalize=True, dropna=True)
    if counts.empty:
        return float("nan")
    return float(counts.iloc[0])


def _series_key(series: pd.Series) -> tuple[Any, ...]:
    return tuple(None if pd.isna(v) else float(v) for v in series.to_list())


def _join_flags(flags: list[str]) -> str:
    return " | ".join(dict.fromkeys(f for f in flags if f))


def _status_from_flags(flags: list[str]) -> str:
    critical_markers = {
        "all_missing",
        "insufficient_non_missing",
        "constant",
        "complete_case_constant",
        "exact_duplicate",
        "severe_vif",
        "severe_pairwise_corr",
        "critical_condition_cluster",
    }
    warning_markers = {
        "near_constant",
        "high_vif",
        "high_pairwise_corr",
        "warning_condition_cluster",
    }
    if any(flag in critical_markers for flag in flags):
        return "critical"
    if any(flag in warning_markers for flag in flags):
        return "warning"
    return "ok"


def _status_rank(status: str) -> int:
    return {"critical": 0, "warning": 1, "ok": 2}.get(status, 3)


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


def _max_finite_or_inf(values: list[float]) -> float:
    if not values:
        return float("nan")
    if any(np.isposinf(v) for v in values):
        return float("inf")
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(max(finite))


def assess_multicollinearity(
    df: pd.DataFrame,
    features: list[str],
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    structural_columns = [
        "feature",
        "non_missing_count",
        "missing_count",
        "missing_rate",
        "unique_values",
        "std",
        "relative_std",
        "dominant_share",
        "duplicate_of",
        "included_in_matrix",
        "status",
        "flags",
    ]
    vif_columns = [
        "feature",
        "vif",
        "tolerance",
        "max_abs_pairwise_corr",
        "most_correlated_with",
        "missing_rate",
        "unique_values",
        "included_in_matrix",
        "condition_cluster_count",
        "status",
        "flags",
        "condition_number",
    ]
    pairwise_columns = ["feature_a", "feature_b", "corr", "abs_corr", "direction", "severity"]
    condition_columns = [
        "dimension",
        "eigenvalue",
        "variance_share",
        "condition_index",
        "contributing_feature_count",
        "contributing_features",
        "severity",
    ]
    variance_columns = ["dimension", "feature", "variance_proportion", "condition_index", "severity"]

    x = _safe_numeric_frame(df, features)
    n_rows_total = int(len(x))

    if not features:
        return {
            "summary": {
                "status": "ok",
                "headline": "No features supplied for multicollinearity assessment",
                "n_rows_total": 0,
                "n_features_requested": 0,
                "n_features_analyzed": 0,
                "n_complete_cases": 0,
                "complete_case_rate": None,
                "matrix_rank": 0,
                "full_rank": True,
                "max_vif": None,
                "max_abs_pairwise_corr": None,
                "max_condition_index": None,
                "exact_duplicate_features": 0,
                "constant_features": 0,
                "near_constant_features": 0,
                "high_corr_pairs": 0,
                "severe_corr_pairs": 0,
                "high_vif_features": 0,
                "severe_vif_features": 0,
                "warning_condition_dimensions": 0,
                "critical_condition_dimensions": 0,
                "flagged_feature_count": 0,
                "flagged_features": "",
                "recommended_actions": "None",
            },
            "structural": pd.DataFrame(columns=structural_columns),
            "vif": pd.DataFrame(columns=vif_columns),
            "pairwise": pd.DataFrame(columns=pairwise_columns),
            "condition": pd.DataFrame(columns=condition_columns),
            "variance_decomposition": pd.DataFrame(columns=variance_columns),
        }

    feature_meta: dict[str, dict[str, Any]] = {}
    for col in features:
        series = x[col]
        non_missing = series.dropna()
        non_missing_count = int(non_missing.shape[0])
        missing_count = int(series.isna().sum())
        unique_values = int(non_missing.nunique()) if non_missing_count else 0
        std = float(non_missing.std(ddof=0)) if non_missing_count else float("nan")
        mean_abs = float(non_missing.abs().mean()) if non_missing_count else float("nan")
        relative_std = float(std / max(mean_abs, ZERO_STD_THRESHOLD)) if np.isfinite(std) else float("nan")
        dominant_share = _dominant_share(series)
        flags: list[str] = []

        if non_missing_count == 0:
            flags.append("all_missing")
        elif non_missing_count < MIN_NON_MISSING:
            flags.append("insufficient_non_missing")
        elif unique_values <= 1 or (np.isfinite(std) and std <= ZERO_STD_THRESHOLD):
            flags.append("constant")
        else:
            if (
                (np.isfinite(std) and std <= ABSOLUTE_NEAR_CONSTANT_STD_THRESHOLD)
                or (np.isfinite(relative_std) and relative_std <= RELATIVE_NEAR_CONSTANT_STD_THRESHOLD)
                or (np.isfinite(dominant_share) and dominant_share >= DOMINANT_SHARE_THRESHOLD)
            ):
                flags.append("near_constant")

        feature_meta[col] = {
            "feature": col,
            "non_missing_count": non_missing_count,
            "missing_count": missing_count,
            "missing_rate": float(missing_count / n_rows_total) if n_rows_total else float("nan"),
            "unique_values": unique_values,
            "std": std,
            "relative_std": relative_std,
            "dominant_share": dominant_share,
            "duplicate_of": "",
            "included_in_matrix": False,
            "flags": flags,
        }

    seen_keys: dict[tuple[Any, ...], str] = {}
    for col in features:
        key = _series_key(x[col])
        if key in seen_keys:
            feature_meta[col]["duplicate_of"] = seen_keys[key]
            feature_meta[col]["flags"].append("exact_duplicate")
        else:
            seen_keys[key] = col

    excluded_structural_flags = {"all_missing", "insufficient_non_missing", "constant"}
    analysis_features = [col for col in features if not excluded_structural_flags.intersection(feature_meta[col]["flags"])]

    complete_cases = x[analysis_features].dropna().copy() if analysis_features else pd.DataFrame()
    while analysis_features and not complete_cases.empty:
        stds = complete_cases[analysis_features].std(ddof=0)
        dropped = [col for col in analysis_features if not np.isfinite(stds[col]) or float(stds[col]) <= ZERO_STD_THRESHOLD]
        if not dropped:
            break
        for col in dropped:
            feature_meta[col]["flags"].append("complete_case_constant")
        analysis_features = [col for col in analysis_features if col not in dropped]
        complete_cases = x[analysis_features].dropna().copy() if analysis_features else pd.DataFrame()

    for col in analysis_features:
        feature_meta[col]["included_in_matrix"] = True

    pairwise_rows: list[dict[str, Any]] = []
    feature_pairwise_lookup: dict[str, dict[str, Any]] = {
        col: {"max_abs_pairwise_corr": float("nan"), "most_correlated_with": ""} for col in features
    }
    if len(analysis_features) >= 2 and not complete_cases.empty:
        corr = complete_cases[analysis_features].corr(method="pearson")
        for i, col_a in enumerate(analysis_features):
            for j in range(i + 1, len(analysis_features)):
                col_b = analysis_features[j]
                corr_val = float(corr.loc[col_a, col_b])
                if not np.isfinite(corr_val):
                    continue
                abs_corr = abs(corr_val)
                if abs_corr > feature_pairwise_lookup[col_a]["max_abs_pairwise_corr"] or not np.isfinite(
                    feature_pairwise_lookup[col_a]["max_abs_pairwise_corr"]
                ):
                    feature_pairwise_lookup[col_a] = {
                        "max_abs_pairwise_corr": abs_corr,
                        "most_correlated_with": col_b,
                    }
                if abs_corr > feature_pairwise_lookup[col_b]["max_abs_pairwise_corr"] or not np.isfinite(
                    feature_pairwise_lookup[col_b]["max_abs_pairwise_corr"]
                ):
                    feature_pairwise_lookup[col_b] = {
                        "max_abs_pairwise_corr": abs_corr,
                        "most_correlated_with": col_a,
                    }
                if abs_corr >= PAIRWISE_WARN_THRESHOLD:
                    pairwise_rows.append(
                        {
                            "feature_a": col_a,
                            "feature_b": col_b,
                            "corr": corr_val,
                            "abs_corr": abs_corr,
                            "direction": "positive" if corr_val >= 0 else "negative",
                            "severity": "severe" if abs_corr >= PAIRWISE_SEVERE_THRESHOLD else "warning",
                        }
                    )

    pairwise_df = pd.DataFrame(pairwise_rows, columns=pairwise_columns).sort_values(
        ["abs_corr", "feature_a", "feature_b"], ascending=[False, True, True]
    )

    vif_lookup: dict[str, float] = {}
    if analysis_features and not complete_cases.empty:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        design = sm.add_constant(complete_cases[analysis_features], has_constant="add")
        for i, col in enumerate(design.columns):
            if col == "const":
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    vif_val = float(variance_inflation_factor(design.values, i))
                vif_lookup[col] = vif_val if np.isfinite(vif_val) or np.isposinf(vif_val) else float("inf")
            except Exception:
                vif_lookup[col] = float("inf")

    matrix_rank = 0
    design_condition_number = float("nan")
    condition_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    feature_condition_flags: dict[str, dict[str, Any]] = {
        col: {"warning": 0, "critical": 0, "dimensions": []} for col in features
    }
    if analysis_features and not complete_cases.empty:
        matrix = complete_cases[analysis_features].to_numpy(dtype=float)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scaled = centered / matrix.std(axis=0, ddof=0, keepdims=True)
        matrix_rank = int(np.linalg.matrix_rank(scaled))

        xtx = scaled.T @ scaled
        eigvals, eigvecs = np.linalg.eigh(xtx)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        max_eig = float(eigvals[0]) if eigvals.size else float("nan")
        safe_eigvals = np.maximum(eigvals.astype(float), ZERO_STD_THRESHOLD)
        cond_idx = np.sqrt(max_eig / safe_eigvals) if np.isfinite(max_eig) and max_eig > 0 else np.full_like(safe_eigvals, np.nan)
        design_condition_number = float(np.nanmax(cond_idx)) if cond_idx.size else float("nan")
        total_eig = float(np.sum(eigvals)) if eigvals.size else float("nan")
        phi = (eigvecs ** 2) / safe_eigvals[np.newaxis, :]
        phi_sum = phi.sum(axis=1, keepdims=True)
        phi_sum[phi_sum == 0] = 1.0
        variance_props = phi / phi_sum

        for dim_idx in range(len(analysis_features)):
            contributing = [
                analysis_features[j]
                for j in range(len(analysis_features))
                if variance_props[j, dim_idx] >= VARIANCE_PROP_THRESHOLD
            ]
            severity = "ok"
            if cond_idx[dim_idx] >= CONDITION_SEVERE_THRESHOLD and len(contributing) >= 2:
                severity = "critical"
            elif cond_idx[dim_idx] >= CONDITION_WARN_THRESHOLD and len(contributing) >= 2:
                severity = "warning"
            elif cond_idx[dim_idx] >= CONDITION_SEVERE_THRESHOLD:
                severity = "warning"

            if severity in {"warning", "critical"}:
                for feature in contributing:
                    feature_condition_flags[feature][severity] += 1
                    feature_condition_flags[feature]["dimensions"].append(int(dim_idx + 1))

            condition_rows.append(
                {
                    "dimension": int(dim_idx + 1),
                    "eigenvalue": float(eigvals[dim_idx]),
                    "variance_share": float(eigvals[dim_idx] / total_eig) if total_eig and np.isfinite(total_eig) else float("nan"),
                    "condition_index": float(cond_idx[dim_idx]),
                    "contributing_feature_count": int(len(contributing)),
                    "contributing_features": " | ".join(contributing),
                    "severity": severity,
                }
            )

            for feature_idx, feature in enumerate(analysis_features):
                prop = float(variance_props[feature_idx, dim_idx])
                if cond_idx[dim_idx] >= CONDITION_WARN_THRESHOLD or prop >= VARIANCE_PROP_THRESHOLD:
                    variance_severity = "ok"
                    if cond_idx[dim_idx] >= CONDITION_SEVERE_THRESHOLD and prop >= VARIANCE_PROP_THRESHOLD:
                        variance_severity = "critical"
                    elif cond_idx[dim_idx] >= CONDITION_WARN_THRESHOLD and prop >= VARIANCE_PROP_THRESHOLD:
                        variance_severity = "warning"
                    variance_rows.append(
                        {
                            "dimension": int(dim_idx + 1),
                            "feature": feature,
                            "variance_proportion": prop,
                            "condition_index": float(cond_idx[dim_idx]),
                            "severity": variance_severity,
                        }
                    )

    condition_df = pd.DataFrame(condition_rows, columns=condition_columns).sort_values(
        ["condition_index", "dimension"], ascending=[False, True]
    )
    variance_df = pd.DataFrame(variance_rows, columns=variance_columns).sort_values(
        ["condition_index", "variance_proportion", "feature"], ascending=[False, False, True]
    )

    structural_rows: list[dict[str, Any]] = []
    vif_rows: list[dict[str, Any]] = []
    for col in features:
        base_flags = list(feature_meta[col]["flags"])
        max_abs_corr = feature_pairwise_lookup[col]["max_abs_pairwise_corr"]
        most_correlated_with = feature_pairwise_lookup[col]["most_correlated_with"]
        vif_val = vif_lookup.get(col, float("nan"))
        condition_cluster_count = int(feature_condition_flags[col]["warning"] + feature_condition_flags[col]["critical"])

        feature_flags = list(base_flags)
        if np.isfinite(max_abs_corr):
            if max_abs_corr >= PAIRWISE_SEVERE_THRESHOLD:
                feature_flags.append("severe_pairwise_corr")
            elif max_abs_corr >= PAIRWISE_WARN_THRESHOLD:
                feature_flags.append("high_pairwise_corr")
        if np.isfinite(vif_val):
            if vif_val >= VIF_SEVERE_THRESHOLD:
                feature_flags.append("severe_vif")
            elif vif_val >= VIF_WARN_THRESHOLD:
                feature_flags.append("high_vif")
        elif np.isposinf(vif_val):
            feature_flags.append("severe_vif")

        if feature_condition_flags[col]["critical"] > 0:
            feature_flags.append("critical_condition_cluster")
        elif feature_condition_flags[col]["warning"] > 0:
            feature_flags.append("warning_condition_cluster")

        status = _status_from_flags(feature_flags)
        flags_text = _join_flags(feature_flags)

        structural_rows.append(
            {
                "feature": col,
                "non_missing_count": feature_meta[col]["non_missing_count"],
                "missing_count": feature_meta[col]["missing_count"],
                "missing_rate": feature_meta[col]["missing_rate"],
                "unique_values": feature_meta[col]["unique_values"],
                "std": feature_meta[col]["std"],
                "relative_std": feature_meta[col]["relative_std"],
                "dominant_share": feature_meta[col]["dominant_share"],
                "duplicate_of": feature_meta[col]["duplicate_of"],
                "included_in_matrix": bool(feature_meta[col]["included_in_matrix"]),
                "status": status,
                "flags": flags_text,
            }
        )

        tolerance = float(1.0 / vif_val) if np.isfinite(vif_val) and vif_val != 0 else (0.0 if np.isposinf(vif_val) else float("nan"))
        vif_rows.append(
            {
                "feature": col,
                "vif": vif_val,
                "tolerance": tolerance,
                "max_abs_pairwise_corr": max_abs_corr,
                "most_correlated_with": most_correlated_with,
                "missing_rate": feature_meta[col]["missing_rate"],
                "unique_values": feature_meta[col]["unique_values"],
                "included_in_matrix": bool(feature_meta[col]["included_in_matrix"]),
                "condition_cluster_count": condition_cluster_count,
                "status": status,
                "flags": flags_text,
                "condition_number": design_condition_number,
            }
        )

    structural_df = pd.DataFrame(structural_rows, columns=structural_columns).sort_values(
        ["status", "missing_rate", "feature"],
        ascending=[True, False, True],
        key=lambda s: s.map(_status_rank) if s.name == "status" else s,
    )
    vif_df = pd.DataFrame(vif_rows, columns=vif_columns).sort_values(
        ["status", "vif", "max_abs_pairwise_corr", "feature"],
        ascending=[True, False, False, True],
        key=lambda s: s.map(_status_rank) if s.name == "status" else s,
    )

    exact_duplicate_features = int(sum("exact_duplicate" in feature_meta[col]["flags"] for col in features))
    constant_features = int(sum("constant" in feature_meta[col]["flags"] for col in features))
    near_constant_features = int(sum("near_constant" in feature_meta[col]["flags"] for col in features))
    high_corr_pairs = int(len(pairwise_df))
    severe_corr_pairs = int((pairwise_df["abs_corr"] >= PAIRWISE_SEVERE_THRESHOLD).sum()) if not pairwise_df.empty else 0
    high_vif_features = int(
        sum(1 for col in analysis_features if np.isfinite(vif_lookup.get(col, float("nan"))) and vif_lookup[col] >= VIF_WARN_THRESHOLD)
        + sum(1 for col in analysis_features if np.isposinf(vif_lookup.get(col, float("nan"))))
    )
    severe_vif_features = int(
        sum(1 for col in analysis_features if np.isfinite(vif_lookup.get(col, float("nan"))) and vif_lookup[col] >= VIF_SEVERE_THRESHOLD)
        + sum(1 for col in analysis_features if np.isposinf(vif_lookup.get(col, float("nan"))))
    )
    warning_condition_dimensions = int((condition_df["severity"] == "warning").sum()) if not condition_df.empty else 0
    critical_condition_dimensions = int((condition_df["severity"] == "critical").sum()) if not condition_df.empty else 0
    max_vif = _max_finite_or_inf(list(vif_lookup.values()))
    max_abs_pairwise_corr = (
        float(pairwise_df["abs_corr"].max()) if not pairwise_df.empty else float("nan")
    )
    max_condition_index = (
        float(condition_df["condition_index"].max()) if not condition_df.empty else float("nan")
    )
    flagged_features = [row["feature"] for row in vif_rows if row["status"] != "ok"]

    status = "ok"
    headline = "No material multicollinearity detected"
    if (
        exact_duplicate_features > 0
        or severe_corr_pairs > 0
        or severe_vif_features > 0
        or critical_condition_dimensions > 0
        or (analysis_features and matrix_rank < len(analysis_features))
    ):
        status = "critical"
        headline = "Severe multicollinearity detected"
    elif near_constant_features > 0 or high_corr_pairs > 0 or high_vif_features > 0 or warning_condition_dimensions > 0:
        status = "warning"
        headline = "Moderate multicollinearity risk detected"

    actions: list[str] = []
    if exact_duplicate_features > 0:
        actions.append("Remove one column from each exact duplicate feature set before interpreting GLM coefficients")
    if severe_corr_pairs > 0:
        actions.append("Collapse or residualize severe pairwise feature clusters instead of keeping both raw signals")
    if severe_vif_features > 0 or critical_condition_dimensions > 0:
        actions.append("Do not treat coefficient magnitudes as stable until the flagged collinearity cluster is reparameterized")
    if near_constant_features > 0:
        actions.append("Drop near-constant predictors unless they are required for business rules")
    complete_case_rate = float(len(complete_cases) / n_rows_total) if n_rows_total else float("nan")
    if np.isfinite(complete_case_rate) and complete_case_rate < 0.75:
        actions.append("Review missing-data handling because complete-case diagnostics are using a reduced sample")
    if not actions:
        actions.append("None")

    summary = {
        "status": status,
        "headline": headline,
        "n_rows_total": n_rows_total,
        "n_features_requested": len(features),
        "n_features_analyzed": len(analysis_features),
        "n_complete_cases": int(len(complete_cases)),
        "complete_case_rate": _safe_float(complete_case_rate),
        "matrix_rank": int(matrix_rank),
        "full_rank": bool(not analysis_features or matrix_rank == len(analysis_features)),
        "max_vif": _safe_float(max_vif),
        "max_abs_pairwise_corr": _safe_float(max_abs_pairwise_corr),
        "max_condition_index": _safe_float(max_condition_index),
        "exact_duplicate_features": exact_duplicate_features,
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
        "high_corr_pairs": high_corr_pairs,
        "severe_corr_pairs": severe_corr_pairs,
        "high_vif_features": high_vif_features,
        "severe_vif_features": severe_vif_features,
        "warning_condition_dimensions": warning_condition_dimensions,
        "critical_condition_dimensions": critical_condition_dimensions,
        "flagged_feature_count": len(flagged_features),
        "flagged_features": " | ".join(flagged_features),
        "recommended_actions": " | ".join(actions),
    }

    return {
        "summary": summary,
        "structural": structural_df,
        "vif": vif_df,
        "pairwise": pairwise_df,
        "condition": condition_df,
        "variance_decomposition": variance_df,
    }


def vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return assess_multicollinearity(df, features)["vif"]


def _ordered_games(df: pd.DataFrame, target_col: str = "home_win") -> pd.DataFrame:
    work = df[df[target_col].notna()].copy()
    if "start_time_utc" in work.columns:
        return work.sort_values("start_time_utc")
    if "game_date_utc" in work.columns:
        return work.sort_values("game_date_utc")
    return work.reset_index(drop=True)


def _glm_original_coefficients(model: GLMLogitModel) -> pd.DataFrame:
    coef_scaled = np.asarray(model.model.coef_[0], dtype=float)
    scale = np.asarray(model.scaler.scale_, dtype=float)
    coef_original = np.divide(
        coef_scaled,
        scale,
        out=np.full_like(coef_scaled, np.nan, dtype=float),
        where=np.isfinite(scale) & (np.abs(scale) > 0),
    )
    return pd.DataFrame(
        {
            "feature": list(model.feature_columns),
            "coef_scaled": coef_scaled,
            "coef_original": coef_original,
        }
    )


def _cv_work_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    work = _ordered_games(df, target_col=target_col)
    if "start_time_utc" not in work.columns and "game_date_utc" in work.columns:
        work = work.copy()
        work["start_time_utc"] = pd.to_datetime(work["game_date_utc"])
    return work


def cv_glm_stability_report(
    df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "home_win",
    n_splits: int = 5,
    min_train_size: int | None = None,
    c: float = 1.0,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    work = _cv_work_frame(df, target_col=target_col)
    if work.empty or not features:
        return {
            "summary": {"status": "insufficient_data", "folds_used": 0, "feature_count": 0},
            "fold_metrics": pd.DataFrame(),
            "coefficients": pd.DataFrame(),
            "feature_summary": pd.DataFrame(),
        }

    resolved_min_train = min(220, max(80, len(work) // 2)) if min_train_size is None else int(min_train_size)
    splits = time_series_splits(work, n_splits=max(2, int(n_splits)), min_train_size=resolved_min_train)
    if not splits:
        return {
            "summary": {"status": "insufficient_data", "folds_used": 0, "feature_count": len(features)},
            "fold_metrics": pd.DataFrame(),
            "coefficients": pd.DataFrame(),
            "feature_summary": pd.DataFrame(),
        }

    fold_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr = work.loc[tr_idx].copy().sort_values("start_time_utc")
        va = work.loc[va_idx].copy().sort_values("start_time_utc")
        if tr.empty or va.empty or tr[target_col].nunique() < 2:
            continue

        model = GLMLogitModel(c=float(c))
        model.fit(tr, features, target_col=target_col)
        p = model.predict_proba(va)
        metrics = metric_bundle(va[target_col].astype(int).to_numpy(), p)
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "train_end": str(tr.iloc[-1]["start_time_utc"]),
                "valid_end": str(va.iloc[-1]["start_time_utc"]),
                "log_loss": float(metrics["log_loss"]),
                "brier": float(metrics["brier"]),
                "accuracy": float(metrics["accuracy"]),
                "auc": float(metrics.get("auc", float("nan"))),
            }
        )
        coef_frame = _glm_original_coefficients(model)
        for row in coef_frame.itertuples(index=False):
            coef_rows.append(
                {
                    "fold": int(fold),
                    "train_end": str(tr.iloc[-1]["start_time_utc"]),
                    "valid_end": str(va.iloc[-1]["start_time_utc"]),
                    "feature": row.feature,
                    "coef_scaled": float(row.coef_scaled),
                    "coef_original": float(row.coef_original) if np.isfinite(row.coef_original) else float("nan"),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coef_rows)
    if coefficients.empty:
        feature_summary = pd.DataFrame()
    else:
        feature_summary = (
            coefficients.groupby("feature", as_index=False)
            .agg(
                fold_count=("coef_original", "count"),
                coef_mean=("coef_original", "mean"),
                coef_std=("coef_original", "std"),
                coef_min=("coef_original", "min"),
                coef_max=("coef_original", "max"),
            )
            .reset_index(drop=True)
        )
        p05 = coefficients.groupby("feature", as_index=False)["coef_original"].quantile(0.05).rename(
            columns={"coef_original": "coef_p05"}
        )
        p95 = coefficients.groupby("feature", as_index=False)["coef_original"].quantile(0.95).rename(
            columns={"coef_original": "coef_p95"}
        )
        sign_flip = (
            coefficients.assign(sign=np.sign(coefficients["coef_original"]).replace(0.0, np.nan))
            .groupby("feature")["sign"]
            .agg(lambda s: int(s.dropna().nunique() > 1))
            .rename("sign_flip_flag")
        )
        feature_summary = (
            feature_summary.merge(p05, on="feature", how="left")
            .merge(p95, on="feature", how="left")
            .merge(sign_flip, on="feature", how="left")
            .sort_values("coef_std", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    summary = {
        "status": "ok" if not fold_metrics.empty else "insufficient_data",
        "folds_used": int(len(fold_metrics)),
        "feature_count": int(len(features)),
        "mean_holdout_log_loss": _safe_float(fold_metrics["log_loss"].mean()) if not fold_metrics.empty else None,
        "mean_holdout_brier": _safe_float(fold_metrics["brier"].mean()) if not fold_metrics.empty else None,
        "mean_holdout_auc": _safe_float(fold_metrics["auc"].mean()) if not fold_metrics.empty else None,
        "sign_flip_features": int(feature_summary["sign_flip_flag"].fillna(0).sum()) if not feature_summary.empty else 0,
    }
    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "coefficients": coefficients,
        "feature_summary": feature_summary,
    }


def bootstrap_glm_coefficients(
    df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "home_win",
    n_boot: int = 100,
    c: float = 1.0,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    work = df[df[target_col].notna()].copy()
    if work.empty or not features:
        return {
            "summary": {"status": "insufficient_data", "bootstrap_runs": 0, "feature_count": 0},
            "coefficients": pd.DataFrame(),
            "feature_summary": pd.DataFrame(),
        }

    base_model = GLMLogitModel(c=float(c))
    base_model.fit(work, features, target_col=target_col)
    base_frame = _glm_original_coefficients(base_model).rename(
        columns={"coef_scaled": "base_coef_scaled", "coef_original": "base_coef_original"}
    )

    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for boot in range(1, n_boot + 1):
        ix = rng.choice(len(work), size=len(work), replace=True)
        sample = work.iloc[ix].copy()
        if sample[target_col].nunique() < 2:
            continue
        try:
            model = GLMLogitModel(c=float(c))
            model.fit(sample, features, target_col=target_col)
        except Exception:
            continue
        coef_frame = _glm_original_coefficients(model)
        for row in coef_frame.itertuples(index=False):
            rows.append(
                {
                    "bootstrap_run": int(boot),
                    "feature": row.feature,
                    "coef_scaled": float(row.coef_scaled),
                    "coef_original": float(row.coef_original) if np.isfinite(row.coef_original) else float("nan"),
                }
            )

    coefficients = pd.DataFrame(rows)
    if coefficients.empty:
        feature_summary = pd.DataFrame()
    else:
        feature_summary = (
            coefficients.groupby("feature", as_index=False)
            .agg(
                bootstrap_count=("coef_original", "count"),
                coef_mean=("coef_original", "mean"),
                coef_std=("coef_original", "std"),
            )
            .reset_index(drop=True)
        )
        ci_low = coefficients.groupby("feature", as_index=False)["coef_original"].quantile(0.05).rename(
            columns={"coef_original": "coef_ci_low"}
        )
        ci_high = coefficients.groupby("feature", as_index=False)["coef_original"].quantile(0.95).rename(
            columns={"coef_original": "coef_ci_high"}
        )
        sign_flip = (
            coefficients.assign(sign=np.sign(coefficients["coef_original"]).replace(0.0, np.nan))
            .groupby("feature")["sign"]
            .agg(lambda s: int(s.dropna().nunique() > 1))
            .reset_index(name="sign_flip_flag")
        )
        feature_summary = (
            feature_summary.merge(ci_low, on="feature", how="left")
            .merge(ci_high, on="feature", how="left")
            .merge(base_frame[["feature", "base_coef_original"]], on="feature", how="left")
            .merge(sign_flip, on="feature", how="left")
            .sort_values("coef_std", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    summary = {
        "status": "ok" if not coefficients.empty else "insufficient_data",
        "bootstrap_runs": int(coefficients["bootstrap_run"].nunique()) if not coefficients.empty else 0,
        "feature_count": int(len(features)),
        "sign_flip_features": int(feature_summary["sign_flip_flag"].fillna(0).sum()) if not feature_summary.empty else 0,
    }
    return {
        "summary": summary,
        "coefficients": coefficients,
        "feature_summary": feature_summary,
    }


def break_test_trade_deadline(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "home_win",
    league: str = "NHL",
) -> dict:
    work = df[df[target_col].notna()].copy()
    work["game_date_utc"] = pd.to_datetime(work["game_date_utc"])
    if work.empty:
        return {"delta_coef_l2": float("nan"), "n_pre": 0, "n_post": 0}

    league_code = str(league or "NHL").strip().upper()
    month, day = (2, 1) if league_code == "NBA" else (3, 7)
    deadline = pd.Timestamp(year=work["game_date_utc"].dt.year.max(), month=month, day=day)
    pre = work[work["game_date_utc"] < deadline]
    post = work[work["game_date_utc"] >= deadline]
    if len(pre) < 20 or len(post) < 20:
        return {"delta_coef_l2": float("nan"), "n_pre": len(pre), "n_post": len(post), "deadline": str(deadline.date())}

    pre_x = _safe_numeric_frame(pre, features)
    post_x = _safe_numeric_frame(post, features)
    pre_x = pre_x.fillna(pre_x.median(numeric_only=True)).fillna(0.0)
    post_x = post_x.fillna(post_x.median(numeric_only=True)).fillna(0.0)
    m1 = LogisticRegression(max_iter=1500).fit(pre_x, pre[target_col])
    m2 = LogisticRegression(max_iter=1500).fit(post_x, post[target_col])
    delta = float(np.linalg.norm(m1.coef_[0] - m2.coef_[0]))
    return {"delta_coef_l2": delta, "n_pre": len(pre), "n_post": len(post), "deadline": str(deadline.date())}
    work = df[df[target_col].notna()].copy()
    work["game_date_utc"] = pd.to_datetime(work["game_date_utc"])
    if work.empty:
        return {"delta_coef_l2": float("nan"), "n_pre": 0, "n_post": 0}

    deadline = pd.Timestamp(year=work["game_date_utc"].dt.year.max(), month=3, day=7)
    pre = work[work["game_date_utc"] < deadline]
    post = work[work["game_date_utc"] >= deadline]
    if len(pre) < 20 or len(post) < 20:
        return {"delta_coef_l2": float("nan"), "n_pre": len(pre), "n_post": len(post)}

    m1 = LogisticRegression(max_iter=1500).fit(pre[features], pre[target_col])
    m2 = LogisticRegression(max_iter=1500).fit(post[features], post[target_col])
    delta = float(np.linalg.norm(m1.coef_[0] - m2.coef_[0]))
    return {"delta_coef_l2": delta, "n_pre": len(pre), "n_post": len(post)}
