from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.evaluation.metrics import brier_score

MIN_FEATURE_UNIQUES = 8
MIN_ROWS = 80
DEFAULT_KNOTS = 5
CURVE_GRID_POINTS = 25
HOLDOUT_GAIN_WARN = 0.002
HOLDOUT_GAIN_STRONG = 0.005
SHAPE_GAP_WARN = 0.020
SHAPE_GAP_STRONG = 0.040


def _numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


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


def _evaluate_probs(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score(y, p)),
    }
    if len(np.unique(y)) >= 2:
        out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auc"] = float("nan")
    return out


def _direction_changes(values: np.ndarray) -> int:
    diffs = np.diff(values)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) <= 1:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _status_rank(status: str) -> int:
    return {"strong": 0, "moderate": 1, "weak": 2, "skip": 3}.get(status, 4)


class _LinearPack:
    def __init__(self, medians: pd.Series, scaler: StandardScaler, model: LogisticRegression, features: list[str]):
        self.medians = medians
        self.scaler = scaler
        self.model = model
        self.features = features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = _numeric_frame(df, self.features).fillna(self.medians).fillna(0.0)
        return self.scaler.transform(x.to_numpy(dtype=float))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = self.transform(df)
        return np.clip(self.model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


class _FeatureBasisPack:
    def __init__(
        self,
        *,
        medians: pd.Series,
        scaler: StandardScaler,
        model: LogisticRegression,
        others: list[str],
        feature: str,
        transformer: SplineTransformer,
        label: str,
    ):
        self.medians = medians
        self.scaler = scaler
        self.model = model
        self.others = others
        self.feature = feature
        self.transformer = transformer
        self.label = label

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        work = _numeric_frame(df, self.others + [self.feature]).fillna(self.medians).fillna(0.0)
        others = work[self.others].to_numpy(dtype=float)
        feature_vals = work[[self.feature]].to_numpy(dtype=float)
        basis = self.transformer.transform(feature_vals)
        matrix = np.hstack([others, basis]) if self.others else basis
        return self.scaler.transform(matrix)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = self.transform(df)
        return np.clip(self.model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


def _fit_linear_pack(train_df: pd.DataFrame, features: list[str], target_col: str, c: float) -> _LinearPack:
    work = _numeric_frame(train_df, features)
    medians = work.median(numeric_only=True).fillna(0.0)
    x = work.fillna(medians).fillna(0.0).to_numpy(dtype=float)
    y = train_df[target_col].astype(int).to_numpy()
    scaler = StandardScaler()
    model = LogisticRegression(penalty="l2", C=c, max_iter=4000, solver="lbfgs")
    model.fit(scaler.fit_transform(x), y)
    return _LinearPack(medians=medians, scaler=scaler, model=model, features=features)


def _knot_count(unique_values: int, *, degree: int) -> int:
    if unique_values <= degree + 1:
        return degree + 1
    return max(degree + 1, min(DEFAULT_KNOTS, unique_values - 1))


def _fit_basis_pack(
    train_df: pd.DataFrame,
    *,
    features: list[str],
    focus_feature: str,
    target_col: str,
    c: float,
    degree: int,
    label: str,
) -> _FeatureBasisPack:
    work = _numeric_frame(train_df, features)
    medians = work.median(numeric_only=True).fillna(0.0)
    filled = work.fillna(medians).fillna(0.0)
    others = [feature for feature in features if feature != focus_feature]
    unique_values = int(work[focus_feature].dropna().nunique())
    transformer = SplineTransformer(
        degree=degree,
        n_knots=_knot_count(unique_values, degree=degree),
        knots="quantile",
        include_bias=False,
        extrapolation="linear",
    )
    feature_vals = filled[[focus_feature]].to_numpy(dtype=float)
    basis = transformer.fit_transform(feature_vals)
    other_matrix = filled[others].to_numpy(dtype=float) if others else np.empty((len(filled), 0))
    x = np.hstack([other_matrix, basis]) if others else basis
    y = train_df[target_col].astype(int).to_numpy()
    scaler = StandardScaler()
    model = LogisticRegression(penalty="l2", C=c, max_iter=4000, solver="lbfgs")
    model.fit(scaler.fit_transform(x), y)
    return _FeatureBasisPack(
        medians=medians,
        scaler=scaler,
        model=model,
        others=others,
        feature=focus_feature,
        transformer=transformer,
        label=label,
    )


def _curve_grid(series: pd.Series, grid_points: int) -> np.ndarray:
    non_missing = series.dropna()
    if non_missing.empty:
        return np.array([], dtype=float)
    if non_missing.nunique() <= grid_points:
        return np.sort(non_missing.astype(float).unique())
    quantiles = np.linspace(0.05, 0.95, grid_points)
    return np.unique(np.quantile(non_missing.astype(float).to_numpy(), quantiles))


def assess_nonlinearity(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    features: list[str],
    target_col: str = "home_win",
    c: float = 1.0,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    feature_columns = [
        "feature",
        "status",
        "best_shape",
        "family_hint",
        "holdout_log_loss_gain_best",
        "holdout_brier_gain_best",
        "train_log_loss_gain_best",
        "shape_gap_best",
        "direction_changes_best",
        "recommendation",
        "skip_reason",
        "n_unique",
        "n_non_missing",
        "smooth_holdout_log_loss_gain",
        "hinge_holdout_log_loss_gain",
        "smooth_holdout_brier_gain",
        "hinge_holdout_brier_gain",
        "smooth_shape_gap",
        "hinge_shape_gap",
    ]
    curve_columns = [
        "feature",
        "grid_rank",
        "grid_value",
        "linear_prob",
        "smooth_prob",
        "hinge_prob",
        "smooth_minus_linear",
        "hinge_minus_linear",
    ]

    work_train = train_df[train_df[target_col].notna()].copy()
    work_holdout = holdout_df[holdout_df[target_col].notna()].copy()
    if work_train.empty or work_holdout.empty or not features:
        summary = {
            "status": "insufficient_data",
            "headline": "Non-linearity assessment skipped due to missing train or holdout targets",
            "n_features_requested": len(features),
            "n_features_evaluated": 0,
            "n_features_flagged": 0,
            "top_transform_candidates": "",
            "top_gam_candidates": "",
            "top_mars_candidates": "",
            "recommended_actions": "Need both train and holdout targets to assess non-linearity",
            "coverage_note": "GLMM and DGLM decisions require separate grouped/time-varying diagnostics",
        }
        return {
            "summary": summary,
            "feature_summary": pd.DataFrame(columns=feature_columns),
            "curve_points": pd.DataFrame(columns=curve_columns),
        }

    baseline = _fit_linear_pack(work_train, features=features, target_col=target_col, c=c)
    baseline_train_p = baseline.predict(work_train)
    baseline_holdout_p = baseline.predict(work_holdout)
    baseline_train_metrics = _evaluate_probs(work_train[target_col].astype(int).to_numpy(), baseline_train_p)
    baseline_holdout_metrics = _evaluate_probs(work_holdout[target_col].astype(int).to_numpy(), baseline_holdout_p)

    medians = _numeric_frame(work_train, features).median(numeric_only=True).fillna(0.0)
    feature_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for feature in features:
        feature_series = pd.to_numeric(work_train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_missing = feature_series.dropna()
        n_non_missing = int(non_missing.shape[0])
        n_unique = int(non_missing.nunique()) if n_non_missing else 0

        base_row: dict[str, Any] = {
            "feature": feature,
            "status": "skip",
            "best_shape": "linear",
            "family_hint": "keep_linear",
            "holdout_log_loss_gain_best": 0.0,
            "holdout_brier_gain_best": 0.0,
            "train_log_loss_gain_best": 0.0,
            "shape_gap_best": 0.0,
            "direction_changes_best": 0,
            "recommendation": "keep_linear",
            "skip_reason": "",
            "n_unique": n_unique,
            "n_non_missing": n_non_missing,
            "smooth_holdout_log_loss_gain": 0.0,
            "hinge_holdout_log_loss_gain": 0.0,
            "smooth_holdout_brier_gain": 0.0,
            "hinge_holdout_brier_gain": 0.0,
            "smooth_shape_gap": 0.0,
            "hinge_shape_gap": 0.0,
        }

        if n_non_missing < MIN_ROWS:
            base_row["skip_reason"] = "too_few_non_missing_rows"
            feature_rows.append(base_row)
            continue
        if n_unique < MIN_FEATURE_UNIQUES:
            base_row["skip_reason"] = "too_few_unique_values"
            feature_rows.append(base_row)
            continue

        try:
            smooth_pack = _fit_basis_pack(
                work_train,
                features=features,
                focus_feature=feature,
                target_col=target_col,
                c=c,
                degree=3,
                label="smooth",
            )
            hinge_pack = _fit_basis_pack(
                work_train,
                features=features,
                focus_feature=feature,
                target_col=target_col,
                c=c,
                degree=1,
                label="hinge",
            )
        except Exception as exc:
            base_row["skip_reason"] = f"fit_failed:{type(exc).__name__}"
            feature_rows.append(base_row)
            continue

        smooth_train_p = smooth_pack.predict(work_train)
        smooth_holdout_p = smooth_pack.predict(work_holdout)
        hinge_train_p = hinge_pack.predict(work_train)
        hinge_holdout_p = hinge_pack.predict(work_holdout)

        smooth_train_metrics = _evaluate_probs(work_train[target_col].astype(int).to_numpy(), smooth_train_p)
        smooth_holdout_metrics = _evaluate_probs(work_holdout[target_col].astype(int).to_numpy(), smooth_holdout_p)
        hinge_train_metrics = _evaluate_probs(work_train[target_col].astype(int).to_numpy(), hinge_train_p)
        hinge_holdout_metrics = _evaluate_probs(work_holdout[target_col].astype(int).to_numpy(), hinge_holdout_p)

        grid = _curve_grid(feature_series, CURVE_GRID_POINTS)
        if grid.size:
            base_frame = pd.DataFrame([medians.to_dict()] * len(grid))
            base_frame[feature] = grid
            linear_curve = baseline.predict(base_frame)
            smooth_curve = smooth_pack.predict(base_frame)
            hinge_curve = hinge_pack.predict(base_frame)
        else:
            linear_curve = np.array([], dtype=float)
            smooth_curve = np.array([], dtype=float)
            hinge_curve = np.array([], dtype=float)

        smooth_shape_gap = float(np.max(np.abs(smooth_curve - linear_curve))) if grid.size else 0.0
        hinge_shape_gap = float(np.max(np.abs(hinge_curve - linear_curve))) if grid.size else 0.0
        smooth_gain = float(baseline_holdout_metrics["log_loss"] - smooth_holdout_metrics["log_loss"])
        hinge_gain = float(baseline_holdout_metrics["log_loss"] - hinge_holdout_metrics["log_loss"])
        smooth_brier_gain = float(baseline_holdout_metrics["brier"] - smooth_holdout_metrics["brier"])
        hinge_brier_gain = float(baseline_holdout_metrics["brier"] - hinge_holdout_metrics["brier"])
        smooth_train_gain = float(baseline_train_metrics["log_loss"] - smooth_train_metrics["log_loss"])
        hinge_train_gain = float(baseline_train_metrics["log_loss"] - hinge_train_metrics["log_loss"])

        best_shape = "smooth"
        best_gain = smooth_gain
        best_brier_gain = smooth_brier_gain
        best_train_gain = smooth_train_gain
        best_shape_gap = smooth_shape_gap
        best_direction_changes = _direction_changes(smooth_curve)
        family_hint = "gam"

        if hinge_gain > smooth_gain + 1e-4 or (abs(hinge_gain - smooth_gain) <= 1e-4 and hinge_shape_gap > smooth_shape_gap):
            best_shape = "hinge"
            best_gain = hinge_gain
            best_brier_gain = hinge_brier_gain
            best_train_gain = hinge_train_gain
            best_shape_gap = hinge_shape_gap
            best_direction_changes = _direction_changes(hinge_curve)
            family_hint = "mars"

        status = "weak"
        recommendation = "keep_linear"
        family_output = "keep_linear"
        if best_gain >= HOLDOUT_GAIN_STRONG:
            status = "strong"
        elif best_gain >= HOLDOUT_GAIN_WARN:
            status = "moderate"
        elif best_gain > 0 and best_shape_gap >= SHAPE_GAP_STRONG:
            status = "moderate"

        if status != "weak" and best_train_gain > 0:
            if best_shape == "smooth":
                family_output = "gam"
                if best_direction_changes <= 1:
                    recommendation = "consider_spline_transform_or_gam"
                else:
                    recommendation = "consider_gam"
            else:
                family_output = "mars"
                recommendation = "consider_piecewise_transform_or_mars"
        elif best_train_gain <= 0 and status != "weak":
            status = "weak"

        row = {
            "feature": feature,
            "status": status,
            "best_shape": best_shape if status != "weak" else "linear",
            "family_hint": family_output,
            "holdout_log_loss_gain_best": best_gain if status != "weak" else 0.0,
            "holdout_brier_gain_best": best_brier_gain if status != "weak" else 0.0,
            "train_log_loss_gain_best": best_train_gain if status != "weak" else 0.0,
            "shape_gap_best": best_shape_gap if status != "weak" else 0.0,
            "direction_changes_best": best_direction_changes if status != "weak" else 0,
            "recommendation": recommendation,
            "skip_reason": "",
            "n_unique": n_unique,
            "n_non_missing": n_non_missing,
            "smooth_holdout_log_loss_gain": smooth_gain,
            "hinge_holdout_log_loss_gain": hinge_gain,
            "smooth_holdout_brier_gain": smooth_brier_gain,
            "hinge_holdout_brier_gain": hinge_brier_gain,
            "smooth_shape_gap": smooth_shape_gap,
            "hinge_shape_gap": hinge_shape_gap,
        }
        feature_rows.append(row)

        for idx, grid_value in enumerate(grid):
            curve_rows.append(
                {
                    "feature": feature,
                    "grid_rank": int(idx + 1),
                    "grid_value": float(grid_value),
                    "linear_prob": float(linear_curve[idx]),
                    "smooth_prob": float(smooth_curve[idx]),
                    "hinge_prob": float(hinge_curve[idx]),
                    "smooth_minus_linear": float(smooth_curve[idx] - linear_curve[idx]),
                    "hinge_minus_linear": float(hinge_curve[idx] - linear_curve[idx]),
                }
            )

    feature_df = pd.DataFrame(feature_rows, columns=feature_columns).sort_values(
        ["status", "holdout_log_loss_gain_best", "shape_gap_best", "feature"],
        ascending=[True, False, False, True],
        key=lambda s: s.map(_status_rank) if s.name == "status" else s,
    )
    curve_df = pd.DataFrame(curve_rows, columns=curve_columns).sort_values(["feature", "grid_rank"])

    flagged = feature_df[feature_df["status"].isin(["moderate", "strong"])].copy()
    strong = feature_df[feature_df["status"] == "strong"].copy()
    gam_candidates = flagged[flagged["family_hint"] == "gam"]
    mars_candidates = flagged[flagged["family_hint"] == "mars"]

    actions: list[str] = []
    if not flagged.empty:
        top_flagged = flagged.sort_values(["holdout_log_loss_gain_best", "shape_gap_best"], ascending=[False, False]).head(3)
        actions.append(
            "Review spline or piecewise transforms for: "
            + ", ".join(top_flagged["feature"].astype(str).tolist())
        )
    if len(gam_candidates) >= 3:
        actions.append("Multiple smooth nonlinear predictors are flagged; a GAM-style upgrade is justified")
    if len(mars_candidates) >= 2:
        actions.append("Multiple hinge-like predictors are flagged; a MARS-style model is worth comparing")
    actions.append("GLMM and DGLM choices are not identified by this static shape test alone; evaluate grouped and time-varying structure separately")

    summary = {
        "status": "ok" if flagged.empty else ("strong_signal" if not strong.empty else "moderate_signal"),
        "headline": "No meaningful nonlinear departures detected" if flagged.empty else "Nonlinear signal detected in at least one predictor",
        "baseline_train_log_loss": _safe_float(baseline_train_metrics["log_loss"]),
        "baseline_holdout_log_loss": _safe_float(baseline_holdout_metrics["log_loss"]),
        "baseline_train_brier": _safe_float(baseline_train_metrics["brier"]),
        "baseline_holdout_brier": _safe_float(baseline_holdout_metrics["brier"]),
        "n_features_requested": len(features),
        "n_features_evaluated": int((feature_df["skip_reason"] == "").sum()) if not feature_df.empty else 0,
        "n_features_skipped": int((feature_df["skip_reason"] != "").sum()) if not feature_df.empty else 0,
        "n_features_flagged": int(len(flagged)),
        "n_features_strong": int(len(strong)),
        "top_transform_candidates": " | ".join(
            flagged.sort_values(["holdout_log_loss_gain_best", "shape_gap_best"], ascending=[False, False])
            .head(5)["feature"]
            .astype(str)
            .tolist()
        ),
        "top_gam_candidates": " | ".join(
            gam_candidates.sort_values(["holdout_log_loss_gain_best", "shape_gap_best"], ascending=[False, False])
            .head(5)["feature"]
            .astype(str)
            .tolist()
        ),
        "top_mars_candidates": " | ".join(
            mars_candidates.sort_values(["holdout_log_loss_gain_best", "shape_gap_best"], ascending=[False, False])
            .head(5)["feature"]
            .astype(str)
            .tolist()
        ),
        "recommended_actions": " | ".join(actions),
        "coverage_note": "Targets smooth and hinge-shaped predictor nonlinearity. Random effects and dynamic state structure need separate GLMM/DGLM diagnostics.",
    }

    return {
        "summary": summary,
        "feature_summary": feature_df,
        "curve_points": curve_df,
    }
