from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseProbModel


PROBABILITY_EPS = 1e-6
ACTIVE_COEF_TOLERANCE = 1e-10
DEFAULT_COEFFICIENT_TOP_N = 8
_COMPLEMENT_KIND_ALIASES = {
    "prior": "prior",
    "prior_model": "prior",
    "prior_offset": "prior",
    "market": "market",
    "market_offset": "market",
}
_COMPLEMENT_INPUT_SCALES = {"probability", "logit"}


def normalize_lasso_credibility_kind(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    normalized = _COMPLEMENT_KIND_ALIASES.get(token)
    if normalized is None:
        supported = sorted(_COMPLEMENT_KIND_ALIASES)
        raise ValueError(f"Unsupported lasso-credibility complement kind '{value}'. Valid={supported}")
    return normalized


def default_lasso_credibility_label(kind: str, column: str) -> str:
    normalized = normalize_lasso_credibility_kind(kind)
    if normalized == "market":
        return "Market complement"
    if normalized == "prior":
        return "Prior complement"
    return str(column)


def _clip_probability(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def _sigmoid(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _safe_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Lasso credibility requires feature columns {missing}, but they are missing from the dataframe.")
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _numeric_series(df: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Lasso credibility requires {label} column '{column}', but it is missing from the dataframe.")
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_count = int(values.isna().sum())
    if missing_count:
        raise ValueError(
            f"Lasso credibility requires finite non-null values in {label} column '{column}', "
            f"but found {missing_count} missing or invalid rows."
        )
    return values.astype(float)


def _summary_stats(values: np.ndarray | pd.Series, prefix: str) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {f"{prefix}_count": 0}
    p25, p75 = np.percentile(arr, [25.0, 75.0])
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p75": float(p75),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


class LassoCredibilityModel(BaseProbModel):
    model_name = "lasso_credibility"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        lambda_value: float,
        complement_kind: str,
        complement_column: str,
        complement_label: str | None = None,
        complement_input_scale: str = "probability",
        exposure_column: str | None = None,
        max_iter: int = 500,
        cnvrg_tol: float = 1e-10,
        zero_tol: float = 1e-10,
    ):
        super().__init__()
        if not np.isfinite(lambda_value) or float(lambda_value) <= 0.0:
            raise ValueError("Lasso credibility requires a positive finite lambda_value.")
        input_scale = str(complement_input_scale or "").strip().lower()
        if input_scale not in _COMPLEMENT_INPUT_SCALES:
            raise ValueError(
                f"Unsupported complement_input_scale '{complement_input_scale}'. "
                f"Valid={sorted(_COMPLEMENT_INPUT_SCALES)}"
            )
        self.lambda_value = float(lambda_value)
        self.complement_kind = normalize_lasso_credibility_kind(complement_kind)
        if model_name:
            self.model_name = str(model_name)
        self.complement_column = str(complement_column or "").strip()
        if not self.complement_column:
            raise ValueError("Lasso credibility requires a non-empty complement_column.")
        self.complement_label = str(complement_label or default_lasso_credibility_label(self.complement_kind, self.complement_column))
        self.complement_input_scale = input_scale
        self.exposure_column = str(exposure_column).strip() or None if exposure_column else None
        self.max_iter = int(max_iter)
        self.cnvrg_tol = float(cnvrg_tol)
        self.zero_tol = float(zero_tol)

        self.requested_feature_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.feature_medians: dict[str, float] = {}
        self.feature_means: dict[str, float] = {}
        self.feature_scales: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.model: sm.GLM | None = None
        self.result: Any = None
        self.exog_names: list[str] = []
        self.train_y: np.ndarray | None = None
        self.train_prob: np.ndarray | None = None
        self.intercept_scaled: float = 0.0
        self.intercept_original: float = 0.0
        self.credibility_metadata: dict[str, Any] = {}

    def _resolve_offset(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        values = _numeric_series(df, self.complement_column, label=f"{self.complement_kind} complement")
        if self.complement_input_scale == "probability":
            below_zero = int((values < 0.0).sum())
            above_one = int((values > 1.0).sum())
            if below_zero or above_one:
                raise ValueError(
                    f"Lasso credibility requires probability-scale complement column '{self.complement_column}' "
                    f"to stay within [0, 1], but found {below_zero} rows below 0 and {above_one} rows above 1."
                )
            probability = values.clip(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
            offset = np.log(probability / (1.0 - probability))
            clip_low = int((values <= PROBABILITY_EPS).sum())
            clip_high = int((values >= 1.0 - PROBABILITY_EPS).sum())
        else:
            offset = values.to_numpy(dtype=float)
            probability = pd.Series(_clip_probability(_sigmoid(offset)), index=df.index)
            clip_low = 0
            clip_high = 0

        summary = {
            "input_scale": self.complement_input_scale,
            "offset_scale": "logit",
            "clipped_to_lower_eps_count": clip_low,
            "clipped_to_upper_eps_count": clip_high,
            **_summary_stats(probability.to_numpy(dtype=float), "probability"),
            **_summary_stats(offset, "offset"),
        }
        return np.asarray(offset, dtype=float), probability.to_numpy(dtype=float), summary

    def _resolve_exposure(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.exposure_column:
            unit = np.ones(len(df), dtype=float)
            return unit, {
                "mode": "unit_weight",
                "column": "",
                "zero_exposure_count": 0,
                "positive_exposure_count": int(len(df)),
                "exposure_total": float(np.sum(unit)),
                **_summary_stats(unit, "exposure"),
            }

        values = _numeric_series(df, self.exposure_column, label="exposure")
        negative_count = int((values < 0.0).sum())
        if negative_count:
            raise ValueError(
                f"Lasso credibility requires non-negative exposure values in '{self.exposure_column}', "
                f"but found {negative_count} negative rows."
            )
        exposure = values.to_numpy(dtype=float)
        return exposure, {
            "mode": "column",
            "column": self.exposure_column,
            "zero_exposure_count": int((values == 0.0).sum()),
            "positive_exposure_count": int((values > 0.0).sum()),
            "exposure_total": float(np.sum(exposure)),
            **_summary_stats(exposure, "exposure"),
        }

    def _design_matrix(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        requested = self.requested_feature_columns if fit else list(self.requested_feature_columns)
        numeric = _safe_numeric_frame(df, requested)
        if fit:
            medians = numeric.median(numeric_only=True).fillna(0.0)
            filled = numeric.fillna(medians).fillna(0.0)
            valid_features = [column for column in requested if filled[column].nunique(dropna=False) > 1]
            self.feature_columns = valid_features
            self.feature_medians = {str(column): float(medians.get(column, 0.0)) for column in requested}
            if valid_features:
                scaled = self.scaler.fit_transform(filled[valid_features].to_numpy(dtype=float))
                self.feature_means = {
                    str(column): float(value) for column, value in zip(valid_features, self.scaler.mean_, strict=False)
                }
                self.feature_scales = {
                    str(column): float(value) if float(value) != 0.0 else 1.0
                    for column, value in zip(valid_features, self.scaler.scale_, strict=False)
                }
                frame = pd.DataFrame(scaled, columns=valid_features, index=df.index)
            else:
                self.feature_means = {}
                self.feature_scales = {}
                frame = pd.DataFrame(index=df.index)
        else:
            filled = numeric.fillna(pd.Series(self.feature_medians).reindex(requested)).fillna(0.0)
            if self.feature_columns:
                scaled = self.scaler.transform(filled[self.feature_columns].to_numpy(dtype=float))
                frame = pd.DataFrame(scaled, columns=self.feature_columns, index=df.index)
            else:
                frame = pd.DataFrame(index=df.index)

        design = pd.DataFrame({"const": np.ones(len(df), dtype=float)}, index=df.index)
        if not frame.empty:
            design = pd.concat([design, frame], axis=1)
        return design

    def _coefficient_rows(self) -> list[dict[str, Any]]:
        if self.result is None:
            return []
        params = pd.Series(np.asarray(self.result.params, dtype=float), index=self.exog_names)
        rows: list[dict[str, Any]] = []
        for feature in self.feature_columns:
            coef_scaled = float(params.get(feature, 0.0))
            scale = float(self.feature_scales.get(feature, 1.0)) or 1.0
            coef_original = float(coef_scaled / scale)
            rows.append(
                {
                    "feature": feature,
                    "coef_scaled": coef_scaled,
                    "coef_original": coef_original,
                    "abs_coef_scaled": float(abs(coef_scaled)),
                    "abs_coef_original": float(abs(coef_original)),
                    "sign": int(np.sign(coef_scaled)),
                    "odds_ratio": float(np.exp(coef_original)),
                    "feature_mean": float(self.feature_means.get(feature, 0.0)),
                    "mean": float(self.feature_means.get(feature, 0.0)),
                    "scale": scale,
                    "active": bool(abs(coef_scaled) > ACTIVE_COEF_TOLERANCE),
                }
            )
        return rows

    def lambda_summary(self) -> dict[str, Any]:
        return {
            "penalty_family": "lasso",
            "lambda_value": float(self.lambda_value),
            "c_value": float(1.0 / self.lambda_value),
            "l1_ratio": None,
            "intercept_penalized": False,
            "regularized_refit": False,
            "p_values_reported": False,
        }

    def active_coefficient_summary(self, top_n: int = DEFAULT_COEFFICIENT_TOP_N) -> list[dict[str, Any]]:
        frame = self.coef_frame()
        if frame.empty:
            return []
        active = frame[frame["active"]].head(max(int(top_n), 0))
        rows: list[dict[str, Any]] = []
        for row in active.itertuples(index=False):
            rows.append(
                {
                    "feature": str(row.feature),
                    "coef_scaled": float(row.coef_scaled),
                    "coef_original": float(row.coef_original),
                    "abs_coef_scaled": float(row.abs_coef_scaled),
                    "abs_coef_original": float(row.abs_coef_original),
                    "odds_ratio": float(row.odds_ratio),
                    "sign": int(row.sign),
                    "active_rank": None if pd.isna(row.active_rank) else int(row.active_rank),
                }
            )
        return rows

    def coefficient_path_metadata(self, top_n: int = DEFAULT_COEFFICIENT_TOP_N) -> dict[str, Any]:
        frame = self.coef_frame()
        return {
            "path_columns": [str(column) for column in frame.columns],
            "sort_key": "abs_coef_scaled_desc",
            "top_feature_names": [str(value) for value in frame.head(max(int(top_n), 0))["feature"].tolist()],
            "active_feature_names": [str(value) for value in frame[frame["active"]]["feature"].tolist()],
            "parameterization": self.lambda_summary(),
        }

    def _capture_fit_metadata(
        self,
        *,
        complement_summary: dict[str, Any],
        exposure_summary: dict[str, Any],
    ) -> None:
        params = pd.Series(np.asarray(self.result.params, dtype=float), index=self.exog_names)
        self.intercept_scaled = float(params.get("const", 0.0))
        intercept_adjustment = 0.0
        for feature in self.feature_columns:
            mean_value = float(self.feature_means.get(feature, 0.0))
            scale_value = float(self.feature_scales.get(feature, 1.0)) or 1.0
            intercept_adjustment += float(params.get(feature, 0.0)) * (mean_value / scale_value)
        self.intercept_original = float(self.intercept_scaled - intercept_adjustment)

        coef_frame = self.coef_frame()
        active = coef_frame[coef_frame["active"]].copy() if not coef_frame.empty else pd.DataFrame()
        positive = active[active["coef_original"] > 0.0].sort_values("coef_original", ascending=False)
        negative = active[active["coef_original"] < 0.0].sort_values("coef_original", ascending=True)
        lambda_summary = self.lambda_summary()

        def _relativity_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
            if frame.empty:
                return []
            return [
                {
                    "feature": str(row["feature"]),
                    "coef_scaled": float(row["coef_scaled"]),
                    "coef_original": float(row["coef_original"]),
                    "odds_ratio": float(row["odds_ratio"]),
                }
                for _, row in frame.head(5).iterrows()
            ]

        self.credibility_metadata = {
            "complement_kind": self.complement_kind,
            "complement_label": self.complement_label,
            "complement_column": self.complement_column,
            "offset_scale": "logit",
            "driver_features": list(self.feature_columns),
            "complement_summary": dict(complement_summary),
            "relativity_summary": {
                "requested_driver_feature_count": int(len(self.requested_feature_columns)),
                "driver_feature_count": int(len(self.feature_columns)),
                "active_driver_feature_count": int(len(active)),
                "inactive_driver_feature_count": int(max(len(self.feature_columns) - len(active), 0)),
                "coefficient_columns": [str(column) for column in coef_frame.columns],
                "coefficient_path_metadata": self.coefficient_path_metadata(),
                "active_coefficient_summary": self.active_coefficient_summary(),
                "top_positive_relativities": _relativity_rows(positive),
                "top_negative_relativities": _relativity_rows(negative),
            },
            "exposure_summary": dict(exposure_summary),
            "lambda_summary": dict(lambda_summary),
            "lambda_choice_note": (
                f"Fixed lasso credibility penalty lambda={self.lambda_value:.6g}; "
                "the fit stays regularized with no unpenalized refit and no p-values."
            ),
            "complement_choice_note": (
                f"Used '{self.complement_column}' as a fixed {self.complement_kind} complement on the logit scale; "
                "driver coefficients represent residual credibility adjustments around that complement."
            ),
            "p_values_reported": False,
            "p_values_behavior": (
                "L1-regularized GLM is fit with refit=False; coefficient p-values are intentionally not computed or reported."
            ),
        }

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        if target_col not in df.columns:
            raise ValueError(f"Lasso credibility requires target column '{target_col}', but it is missing from the dataframe.")
        work = df[df[target_col].notna()].copy()
        if work.empty:
            raise ValueError("Lasso credibility requires at least one non-null target row to fit.")

        self.requested_feature_columns = list(feature_columns)
        y = work[target_col].astype(int).to_numpy()
        offset, _, complement_summary = self._resolve_offset(work)
        _, exposure_summary = self._resolve_exposure(work)
        design = self._design_matrix(work, fit=True)
        self.exog_names = list(design.columns)

        alpha = np.full(design.shape[1], self.lambda_value, dtype=float)
        alpha[0] = 0.0
        self.model = sm.GLM(y, design, family=sm.families.Binomial(), offset=offset)
        self.result = self.model.fit_regularized(
            alpha=alpha,
            L1_wt=1.0,
            maxiter=self.max_iter,
            cnvrg_tol=self.cnvrg_tol,
            zero_tol=self.zero_tol,
            refit=False,
        )
        self.train_y = y
        self.train_prob = _clip_probability(np.asarray(self.result.predict(exog=design, offset=offset), dtype=float))
        self._capture_fit_metadata(
            complement_summary=complement_summary,
            exposure_summary=exposure_summary,
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Lasso credibility has not been fit.")
        offset, _, _ = self._resolve_offset(df)
        design = self._design_matrix(df, fit=False)
        pred = self.result.predict(exog=design, offset=offset)
        return _clip_probability(np.asarray(pred, dtype=float))

    def coef_frame(self) -> pd.DataFrame:
        rows = self._coefficient_rows()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "coef_scaled",
                    "coef_original",
                    "abs_coef_scaled",
                    "abs_coef_original",
                    "sign",
                    "odds_ratio",
                    "feature_mean",
                    "mean",
                    "scale",
                    "active",
                    "active_rank",
                ]
            )
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(["abs_coef_scaled", "feature"], ascending=[False, True], kind="mergesort").reset_index(
            drop=True
        )
        active_rank = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        if bool(frame["active"].any()):
            active_rank.loc[frame["active"]] = np.arange(1, int(frame["active"].sum()) + 1, dtype=int)
        frame["active_rank"] = active_rank
        return frame

    def fit_summary(self) -> dict[str, Any]:
        if self.train_y is None or self.train_prob is None or self.result is None:
            raise RuntimeError("Lasso credibility has not been fit.")
        y = np.asarray(self.train_y, dtype=int)
        p = _clip_probability(self.train_prob)
        coefficient_frame = self.coef_frame()
        active_features = int((coefficient_frame["active"]).sum()) if self.feature_columns else 0
        active_feature_names = [
            str(value) for value in coefficient_frame[coefficient_frame["active"]]["feature"].tolist()
        ] if not coefficient_frame.empty else []
        active_parameter_count = active_features + int(abs(self.intercept_scaled) > ACTIVE_COEF_TOLERANCE)
        log_likelihood = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        return {
            "fit_variant": f"{self.complement_kind}_offset_lasso_credibility",
            "distribution": "binomial",
            "link_function": "logit",
            "offset_scale": "logit",
            "regularized_refit": False,
            "p_values_reported": False,
            "p_values_behavior": (
                "L1-regularized GLM is fit with refit=False; coefficient p-values are intentionally not computed or reported."
            ),
            "n_obs": int(len(y)),
            "requested_feature_count": int(len(self.requested_feature_columns)),
            "driver_feature_count": int(len(self.feature_columns)),
            "active_feature_count": active_features,
            "active_parameter_count": active_parameter_count,
            "active_features": active_feature_names,
            "coefficient_columns": [str(column) for column in coefficient_frame.columns],
            "active_coefficient_summary": self.active_coefficient_summary(),
            "coefficient_path_metadata": self.coefficient_path_metadata(),
            "lambda_value": float(self.lambda_value),
            "lambda_summary": self.lambda_summary(),
            "intercept_scaled": float(self.intercept_scaled),
            "intercept_original": float(self.intercept_original),
            "train_log_likelihood": log_likelihood,
            "train_deviance": float(-2.0 * log_likelihood),
            "train_log_loss": float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))),
            "complement_kind": self.complement_kind,
            "complement_column": self.complement_column,
        }


__all__ = [
    "ACTIVE_COEF_TOLERANCE",
    "LassoCredibilityModel",
    "PROBABILITY_EPS",
    "default_lasso_credibility_label",
    "normalize_lasso_credibility_kind",
]
