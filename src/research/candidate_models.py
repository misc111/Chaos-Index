from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

PROBABILITY_EPS = 1e-6


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _safe_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _median_series(df: pd.DataFrame, features: list[str]) -> pd.Series:
    return _safe_numeric_frame(df, features).median(numeric_only=True).fillna(0.0)


def _filled_numeric_frame(df: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    frame = _safe_numeric_frame(df, features)
    return frame.fillna(medians.reindex(features)).fillna(0.0)


def _bic_from_loglike(log_likelihood: float, parameter_count: int, n_obs: int) -> float:
    return float(-2.0 * log_likelihood + parameter_count * np.log(max(int(n_obs), 1)))


@dataclass(slots=True)
class CandidateFitStats:
    model_name: str
    display_name: str
    parameter_count: int
    active_parameter_count: int
    n_features: int
    train_log_likelihood: float | None = None
    train_deviance: float | None = None
    train_aic: float | None = None
    train_bic: float | None = None
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "parameter_count": int(self.parameter_count),
            "active_parameter_count": int(self.active_parameter_count),
            "n_features": int(self.n_features),
            "train_log_likelihood": self.train_log_likelihood,
            "train_deviance": self.train_deviance,
            "train_aic": self.train_aic,
            "train_bic": self.train_bic,
            "notes": self.notes,
        }


class BaseCandidateModel:
    model_name = "base_candidate"
    display_name = "Base Candidate"

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit_statistics(self) -> CandidateFitStats:
        raise NotImplementedError


class PenalizedLogitCandidate(BaseCandidateModel):
    def __init__(
        self,
        *,
        model_name: str,
        display_name: str,
        features: list[str],
        penalty: str | None,
        c: float = 1.0,
        l1_ratio: float | None = None,
        solver: str = "lbfgs",
        max_iter: int = 5000,
    ):
        self.model_name = model_name
        self.display_name = display_name
        self.features = list(features)
        self.penalty = penalty
        self.c = float(c)
        self.l1_ratio = None if l1_ratio is None else float(l1_ratio)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.medians = pd.Series(dtype=float)
        self.scaler = StandardScaler()
        self.model: LogisticRegression | None = None
        self.train_y: np.ndarray | None = None
        self.train_prob: np.ndarray | None = None

    def _transform(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        work = _safe_numeric_frame(df, self.features)
        if fit:
            self.medians = work.median(numeric_only=True).fillna(0.0)
        filled = work.fillna(self.medians.reindex(self.features)).fillna(0.0)
        matrix = filled.to_numpy(dtype=float)
        if fit:
            return self.scaler.fit_transform(matrix)
        return self.scaler.transform(matrix)

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        work = df[df[target_col].notna()].copy()
        y = work[target_col].astype(int).to_numpy()
        x = self._transform(work, fit=True)
        kwargs: dict[str, Any] = {
            "max_iter": self.max_iter,
            "solver": self.solver,
            "penalty": self.penalty,
        }
        if self.penalty is not None:
            kwargs["C"] = self.c
        if self.penalty == "elasticnet":
            kwargs["l1_ratio"] = self.l1_ratio
        self.model = LogisticRegression(**kwargs)
        self.model.fit(x, y)
        self.train_y = y
        self.train_prob = _clip_probability(self.model.predict_proba(x)[:, 1])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.model_name} has not been fit")
        return _clip_probability(self.model.predict_proba(self._transform(df, fit=False))[:, 1])

    def fit_statistics(self) -> CandidateFitStats:
        if self.model is None or self.train_y is None or self.train_prob is None:
            raise RuntimeError(f"{self.model_name} has not been fit")
        coefficients = np.asarray(self.model.coef_[0], dtype=float)
        intercept = float(np.asarray(self.model.intercept_, dtype=float)[0])
        log_likelihood = float(
            np.sum(
                self.train_y * np.log(self.train_prob)
                + (1.0 - self.train_y) * np.log(1.0 - self.train_prob)
            )
        )
        parameter_count = int(len(coefficients) + 1)
        active_count = int(np.sum(np.abs(coefficients) > 1e-8) + (abs(intercept) > 1e-8))
        deviance = float(-2.0 * log_likelihood)
        return CandidateFitStats(
            model_name=self.model_name,
            display_name=self.display_name,
            parameter_count=parameter_count,
            active_parameter_count=active_count,
            n_features=len(self.features),
            train_log_likelihood=log_likelihood,
            train_deviance=deviance,
            train_aic=float("nan"),
            train_bic=float("nan"),
            notes=f"penalty={self.penalty or 'none'}; solver={self.solver}",
        )


class VanillaGLMBinomialCandidate(BaseCandidateModel):
    def __init__(self, *, features: list[str]):
        self.features = list(features)
        self.model_name = "glm_vanilla"
        self.display_name = "Vanilla GLM"
        self.medians = pd.Series(dtype=float)
        self.scaler = StandardScaler()
        self.model: sm.GLM | None = None
        self.result: Any = None
        self.exog_names: list[str] = []
        self.train_y: np.ndarray | None = None
        self.train_prob: np.ndarray | None = None

    def _design_matrix(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        numeric = _safe_numeric_frame(df, self.features)
        if fit:
            self.medians = numeric.median(numeric_only=True).fillna(0.0)
        filled = numeric.fillna(self.medians.reindex(self.features)).fillna(0.0)
        values = filled.to_numpy(dtype=float)
        scaled = self.scaler.fit_transform(values) if fit else self.scaler.transform(values)
        frame = pd.DataFrame(scaled, columns=self.features, index=df.index)
        design = sm.add_constant(frame, has_constant="add")
        return design

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        work = df[df[target_col].notna()].copy()
        y = work[target_col].astype(int).to_numpy()
        design = self._design_matrix(work, fit=True)
        self.model = sm.GLM(y, design, family=sm.families.Binomial())
        self.result = self.model.fit()
        self.exog_names = list(design.columns)
        self.train_y = y
        self.train_prob = _clip_probability(np.asarray(self.result.predict(design), dtype=float))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("glm_vanilla has not been fit")
        design = self._design_matrix(df, fit=False).reindex(columns=self.exog_names, fill_value=1.0)
        return _clip_probability(np.asarray(self.result.predict(design), dtype=float))

    def fit_statistics(self) -> CandidateFitStats:
        if self.result is None or self.train_y is None:
            raise RuntimeError("glm_vanilla has not been fit")
        parameter_count = int(len(self.exog_names))
        coefficients = np.asarray(self.result.params, dtype=float)
        active = int(np.sum(np.abs(coefficients) > 1e-8))
        log_likelihood = float(self.result.llf)
        return CandidateFitStats(
            model_name=self.model_name,
            display_name=self.display_name,
            parameter_count=parameter_count,
            active_parameter_count=active,
            n_features=len(self.features),
            train_log_likelihood=log_likelihood,
            train_deviance=float(self.result.deviance),
            train_aic=float(self.result.aic),
            train_bic=_bic_from_loglike(log_likelihood, parameter_count, len(self.train_y)),
            notes="statsmodels.GLM(Binomial)",
        )


class GLMMLogitCandidate(BaseCandidateModel):
    def __init__(
        self,
        *,
        fixed_features: list[str],
        home_group_col: str = "home_team",
        away_group_col: str = "away_team",
    ):
        self.fixed_features = list(fixed_features)
        self.home_group_col = home_group_col
        self.away_group_col = away_group_col
        self.model_name = "glmm_logit"
        self.display_name = "GLMM Logit"
        self.medians = pd.Series(dtype=float)
        self.scaler = StandardScaler()
        self.fixed_feature_names = [f"fx_{idx:02d}" for idx in range(len(self.fixed_features))]
        self.formula = "home_win ~ 1"
        self.vc_formulas = {
            self.home_group_col: f"0 + C({self.home_group_col})",
            self.away_group_col: f"0 + C({self.away_group_col})",
        }
        self.model: BinomialBayesMixedGLM | None = None
        self.result: Any = None
        self.design_info: Any = None
        self.fixed_effect_mean: np.ndarray | None = None
        self.random_effect_mean: np.ndarray | None = None
        self.vcp_mean: np.ndarray | None = None
        self.vc_component_columns: dict[str, list[str]] = {}
        self.fit_method = ""

    def _fit_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df[[self.home_group_col, self.away_group_col]].copy()
        matrix = _filled_numeric_frame(df, self.fixed_features, self.medians)
        scaled = self.scaler.transform(matrix.to_numpy(dtype=float))
        for idx, feature_name in enumerate(self.fixed_feature_names):
            work[feature_name] = scaled[:, idx]
        return work

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        work = df[df[target_col].notna()].copy()
        numeric = _safe_numeric_frame(work, self.fixed_features)
        self.medians = numeric.median(numeric_only=True).fillna(0.0)
        filled = numeric.fillna(self.medians.reindex(self.fixed_features)).fillna(0.0)
        scaled = self.scaler.fit_transform(filled.to_numpy(dtype=float))
        frame = work[[target_col, self.home_group_col, self.away_group_col]].copy()
        for idx, feature_name in enumerate(self.fixed_feature_names):
            frame[feature_name] = scaled[:, idx]
        self.formula = f"{target_col} ~ {' + '.join(self.fixed_feature_names)}" if self.fixed_feature_names else f"{target_col} ~ 1"
        self.model = BinomialBayesMixedGLM.from_formula(self.formula, self.vc_formulas, frame)
        try:
            self.result = self.model.fit_map()
            self.fit_method = "laplace_map"
        except Exception:
            self.result = self.model.fit_vb()
            self.fit_method = "variational_bayes"
        self.design_info = self.model.data.design_info
        self.fixed_effect_mean = np.asarray(self.result.params[: self.model.k_fep], dtype=float)
        self.vcp_mean = np.asarray(self.result.params[self.model.k_fep : self.model.k_fep + self.model.k_vcp], dtype=float)
        self.random_effect_mean = np.asarray(self.result.params[-self.model.k_vc :], dtype=float)
        vc_names = list(getattr(self.model, "vc_names", []))
        self.vc_component_columns = {
            self.home_group_col: [name for name in vc_names if name.startswith(f"C({self.home_group_col})")],
            self.away_group_col: [name for name in vc_names if name.startswith(f"C({self.away_group_col})")],
        }

    def _random_design(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        parts: list[sparse.csr_matrix] = []
        for group_col, vc_formula in self.vc_formulas.items():
            design = patsy.dmatrix(vc_formula, frame, return_type="dataframe")
            design = design.reindex(columns=self.vc_component_columns[group_col], fill_value=0.0)
            parts.append(sparse.csr_matrix(design.to_numpy(dtype=float)))
        if not parts:
            return sparse.csr_matrix((len(frame), 0))
        return sparse.hstack(parts, format="csr")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.result is None or self.fixed_effect_mean is None or self.random_effect_mean is None:
            raise RuntimeError("glmm_logit has not been fit")
        frame = self._fit_frame(df)
        fixed_design = patsy.build_design_matrices([self.design_info], frame, return_type="dataframe")[0]
        random_design = self._random_design(frame)
        linear = fixed_design.to_numpy(dtype=float) @ self.fixed_effect_mean
        if random_design.shape[1]:
            linear = linear + np.asarray(random_design @ self.random_effect_mean, dtype=float).ravel()
        return _clip_probability(_sigmoid(linear))

    def fit_statistics(self) -> CandidateFitStats:
        if self.model is None or self.result is None or self.fixed_effect_mean is None:
            raise RuntimeError("glmm_logit has not been fit")
        parameter_count = int(self.model.k_fep + self.model.k_vcp + self.model.k_vc)
        active_fixed = int(np.sum(np.abs(self.fixed_effect_mean) > 1e-8))
        active_random = int(np.sum(np.abs(self.random_effect_mean) > 1e-8)) if self.random_effect_mean is not None else 0
        home_sd = float(np.exp(self.vcp_mean[0])) if self.vcp_mean is not None and len(self.vcp_mean) >= 1 else float("nan")
        away_sd = float(np.exp(self.vcp_mean[1])) if self.vcp_mean is not None and len(self.vcp_mean) >= 2 else float("nan")
        return CandidateFitStats(
            model_name=self.model_name,
            display_name=self.display_name,
            parameter_count=parameter_count,
            active_parameter_count=active_fixed + active_random,
            n_features=len(self.fixed_features),
            train_log_likelihood=None,
            train_deviance=None,
            train_aic=None,
            train_bic=None,
            notes=f"fit={self.fit_method}; home_random_sd={home_sd:.4f}; away_random_sd={away_sd:.4f}",
        )


class DGLMMarginCandidate(BaseCandidateModel):
    def __init__(self, *, features: list[str], iterations: int = 2):
        self.features = list(features)
        self.iterations = int(max(iterations, 1))
        self.model_name = "dglm_margin"
        self.display_name = "DGLM Margin"
        self.medians = pd.Series(dtype=float)
        self.scaler = StandardScaler()
        self.exog_names: list[str] = []
        self.mean_result: Any = None
        self.dispersion_result: Any = None
        self.train_n = 0

    def _design_matrix(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        numeric = _safe_numeric_frame(df, self.features)
        if fit:
            self.medians = numeric.median(numeric_only=True).fillna(0.0)
        filled = numeric.fillna(self.medians.reindex(self.features)).fillna(0.0)
        values = filled.to_numpy(dtype=float)
        scaled = self.scaler.fit_transform(values) if fit else self.scaler.transform(values)
        frame = pd.DataFrame(scaled, columns=self.features, index=df.index)
        design = sm.add_constant(frame, has_constant="add")
        return design

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        work = df[df[target_col].notna()].copy()
        if "home_score" not in work.columns or "away_score" not in work.columns:
            raise ValueError("DGLM margin candidate requires home_score and away_score")
        margin = pd.to_numeric(work["home_score"], errors="coerce") - pd.to_numeric(work["away_score"], errors="coerce")
        if margin.isna().any():
            raise ValueError("DGLM margin candidate requires realized home_score and away_score")

        design = self._design_matrix(work, fit=True)
        self.exog_names = list(design.columns)
        weights = np.ones(len(work), dtype=float)
        for _ in range(self.iterations):
            self.mean_result = sm.GLM(
                margin.to_numpy(dtype=float),
                design,
                family=sm.families.Gaussian(),
                var_weights=np.clip(1.0 / weights, 1e-6, None),
            ).fit()
            mu = np.asarray(self.mean_result.predict(design), dtype=float)
            squared_residual = np.square(margin.to_numpy(dtype=float) - mu) + 1e-6
            self.dispersion_result = sm.GLM(
                squared_residual,
                design,
                family=sm.families.Gamma(link=sm.families.links.log()),
            ).fit()
            weights = np.clip(np.asarray(self.dispersion_result.predict(design), dtype=float), 1e-6, None)
        self.train_n = int(len(work))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_result is None or self.dispersion_result is None:
            raise RuntimeError("dglm_margin has not been fit")
        design = self._design_matrix(df, fit=False).reindex(columns=self.exog_names, fill_value=1.0)
        mean_margin = np.asarray(self.mean_result.predict(design), dtype=float)
        variance = np.clip(np.asarray(self.dispersion_result.predict(design), dtype=float), 1e-6, None)
        probability = 1.0 - norm.cdf((0.0 - mean_margin) / np.sqrt(variance))
        return _clip_probability(probability)

    def fit_statistics(self) -> CandidateFitStats:
        if self.mean_result is None or self.dispersion_result is None:
            raise RuntimeError("dglm_margin has not been fit")
        parameter_count = int(len(self.exog_names) * 2)
        active_parameters = int(
            np.sum(np.abs(np.asarray(self.mean_result.params, dtype=float)) > 1e-8)
            + np.sum(np.abs(np.asarray(self.dispersion_result.params, dtype=float)) > 1e-8)
        )
        mean_log_like = float(getattr(self.mean_result, "llf", np.nan))
        dispersion_log_like = float(getattr(self.dispersion_result, "llf", np.nan))
        total_log_like = mean_log_like + dispersion_log_like if np.isfinite(mean_log_like) and np.isfinite(dispersion_log_like) else None
        mean_deviance = float(getattr(self.mean_result, "deviance", np.nan))
        dispersion_deviance = float(getattr(self.dispersion_result, "deviance", np.nan))
        total_deviance = mean_deviance + dispersion_deviance if np.isfinite(mean_deviance) and np.isfinite(dispersion_deviance) else None
        total_aic = float(self.mean_result.aic + self.dispersion_result.aic)
        total_bic = None
        if total_log_like is not None:
            total_bic = _bic_from_loglike(total_log_like, parameter_count, self.train_n)
        return CandidateFitStats(
            model_name=self.model_name,
            display_name=self.display_name,
            parameter_count=parameter_count,
            active_parameter_count=active_parameters,
            n_features=len(self.features),
            train_log_likelihood=total_log_like,
            train_deviance=total_deviance,
            train_aic=total_aic,
            train_bic=total_bic,
            notes=f"iterations={self.iterations}",
        )


class GAMSplineCandidate(BaseCandidateModel):
    def __init__(
        self,
        *,
        linear_features: list[str],
        spline_features: list[str],
        c: float = 1.0,
        n_knots: int = 5,
    ):
        self.linear_features = list(dict.fromkeys(linear_features))
        self.spline_features = [feature for feature in dict.fromkeys(spline_features) if feature not in self.linear_features]
        self.c = float(c)
        self.n_knots = int(max(n_knots, 4))
        self.model_name = "gam_spline"
        self.display_name = "GAM Spline"
        self.medians = pd.Series(dtype=float)
        self.linear_scaler = StandardScaler()
        self.basis_scaler = StandardScaler()
        self.linear_transformers: dict[str, SplineTransformer] = {}
        self.model: LogisticRegression | None = None
        self.basis_feature_count = 0
        self.train_y: np.ndarray | None = None
        self.train_prob: np.ndarray | None = None

    def _all_features(self) -> list[str]:
        return list(dict.fromkeys(self.linear_features + self.spline_features))

    def _build_matrix(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        all_features = self._all_features()
        numeric = _safe_numeric_frame(df, all_features)
        if fit:
            self.medians = numeric.median(numeric_only=True).fillna(0.0)
        filled = numeric.fillna(self.medians.reindex(all_features)).fillna(0.0)
        parts: list[np.ndarray] = []
        if self.linear_features:
            parts.append(filled[self.linear_features].to_numpy(dtype=float))
        spline_parts: list[np.ndarray] = []
        for feature in self.spline_features:
            values = filled[[feature]].to_numpy(dtype=float)
            if fit:
                transformer = SplineTransformer(
                    degree=3,
                    n_knots=self.n_knots,
                    knots="quantile",
                    include_bias=False,
                    extrapolation="linear",
                )
                spline_values = transformer.fit_transform(values)
                self.linear_transformers[feature] = transformer
            else:
                spline_values = self.linear_transformers[feature].transform(values)
            spline_parts.append(spline_values)
        if spline_parts:
            parts.append(np.hstack(spline_parts))
        matrix = np.hstack(parts) if parts else np.empty((len(df), 0))
        if fit:
            self.basis_feature_count = int(matrix.shape[1])
            return self.basis_scaler.fit_transform(matrix)
        return self.basis_scaler.transform(matrix)

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        work = df[df[target_col].notna()].copy()
        y = work[target_col].astype(int).to_numpy()
        x = self._build_matrix(work, fit=True)
        self.model = LogisticRegression(max_iter=5000, penalty="l2", C=self.c, solver="lbfgs")
        self.model.fit(x, y)
        self.train_y = y
        self.train_prob = _clip_probability(self.model.predict_proba(x)[:, 1])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("gam_spline has not been fit")
        return _clip_probability(self.model.predict_proba(self._build_matrix(df, fit=False))[:, 1])

    def fit_statistics(self) -> CandidateFitStats:
        if self.model is None or self.train_y is None or self.train_prob is None:
            raise RuntimeError("gam_spline has not been fit")
        coefficients = np.asarray(self.model.coef_[0], dtype=float)
        intercept = float(np.asarray(self.model.intercept_, dtype=float)[0])
        log_likelihood = float(
            np.sum(
                self.train_y * np.log(self.train_prob)
                + (1.0 - self.train_y) * np.log(1.0 - self.train_prob)
            )
        )
        parameter_count = int(len(coefficients) + 1)
        active_count = int(np.sum(np.abs(coefficients) > 1e-8) + (abs(intercept) > 1e-8))
        return CandidateFitStats(
            model_name=self.model_name,
            display_name=self.display_name,
            parameter_count=parameter_count,
            active_parameter_count=active_count,
            n_features=len(self._all_features()),
            train_log_likelihood=log_likelihood,
            train_deviance=float(-2.0 * log_likelihood),
            train_aic=float("nan"),
            train_bic=float("nan"),
            notes=f"spline_features={len(self.spline_features)}; n_knots={self.n_knots}",
        )


class MARSHingeCandidate(BaseCandidateModel):
    def __init__(
        self,
        *,
        linear_features: list[str],
        hinge_features: list[str],
        c: float = 0.5,
        knots_per_feature: int = 4,
        interaction_degree: int = 1,
    ):
        self.linear_features = list(dict.fromkeys(linear_features))
        self.hinge_features = [feature for feature in dict.fromkeys(hinge_features) if feature not in self.linear_features]
        self.c = float(c)
        self.knots_per_feature = int(max(knots_per_feature, 2))
        self.interaction_degree = int(max(interaction_degree, 1))
        self.model_name = "mars_hinge"
        self.display_name = "MARS Hinge"
        self.medians = pd.Series(dtype=float)
        self.scaler = StandardScaler()
        self.knots_: dict[str, np.ndarray] = {}
        self.model: LogisticRegression | None = None
        self.basis_names: list[str] = []
        self.train_y: np.ndarray | None = None
        self.train_prob: np.ndarray | None = None

    def _all_features(self) -> list[str]:
        return list(dict.fromkeys(self.linear_features + self.hinge_features))

    def _build_hinge_basis(self, filled: pd.DataFrame, *, fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []
        basis_names: list[str] = []
        if self.linear_features:
            parts.append(filled[self.linear_features].to_numpy(dtype=float))
            basis_names.extend(self.linear_features)

        hinge_cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for feature in self.hinge_features:
            values = filled[feature].to_numpy(dtype=float)
            if fit:
                quantiles = np.linspace(0.15, 0.85, self.knots_per_feature)
                self.knots_[feature] = np.unique(np.quantile(values, quantiles))
            positive_negative_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for knot in self.knots_[feature]:
                positive = np.maximum(values - knot, 0.0)[:, None]
                negative = np.maximum(knot - values, 0.0)[:, None]
                positive_negative_pairs.append((positive, negative))
                parts.extend([positive, negative])
                basis_names.extend([f"{feature}_hinge_pos_{knot:.6f}", f"{feature}_hinge_neg_{knot:.6f}"])
            hinge_cache[feature] = positive_negative_pairs

        if self.interaction_degree >= 2 and len(self.hinge_features) >= 2:
            # Keep interaction count small so the basis stays stable on modest sample sizes.
            for feature_a, feature_b in combinations(self.hinge_features[:3], 2):
                pairs_a = hinge_cache.get(feature_a, [])[:2]
                pairs_b = hinge_cache.get(feature_b, [])[:2]
                for idx_a, (a_pos, a_neg) in enumerate(pairs_a, start=1):
                    for idx_b, (b_pos, b_neg) in enumerate(pairs_b, start=1):
                        interaction_pos = a_pos * b_pos
                        interaction_neg = a_neg * b_neg
                        parts.extend([interaction_pos, interaction_neg])
                        basis_names.extend(
                            [
                                f"{feature_a}_x_{feature_b}_pp_{idx_a}_{idx_b}",
                                f"{feature_a}_x_{feature_b}_nn_{idx_a}_{idx_b}",
                            ]
                        )

        self.basis_names = basis_names
        if not parts:
            return np.empty((len(filled), 0))
        return np.hstack(parts)

    def _matrix(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        all_features = self._all_features()
        numeric = _safe_numeric_frame(df, all_features)
        if fit:
            self.medians = numeric.median(numeric_only=True).fillna(0.0)
        filled = numeric.fillna(self.medians.reindex(all_features)).fillna(0.0)
        basis = self._build_hinge_basis(filled, fit=fit)
        if fit:
            return self.scaler.fit_transform(basis)
        return self.scaler.transform(basis)

    def fit(self, df: pd.DataFrame, *, target_col: str = "home_win") -> None:
        work = df[df[target_col].notna()].copy()
        y = work[target_col].astype(int).to_numpy()
        x = self._matrix(work, fit=True)
        self.model = LogisticRegression(max_iter=5000, penalty="l1", C=self.c, solver="saga")
        self.model.fit(x, y)
        self.train_y = y
        self.train_prob = _clip_probability(self.model.predict_proba(x)[:, 1])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("mars_hinge has not been fit")
        return _clip_probability(self.model.predict_proba(self._matrix(df, fit=False))[:, 1])

    def fit_statistics(self) -> CandidateFitStats:
        if self.model is None or self.train_y is None or self.train_prob is None:
            raise RuntimeError("mars_hinge has not been fit")
        coefficients = np.asarray(self.model.coef_[0], dtype=float)
        intercept = float(np.asarray(self.model.intercept_, dtype=float)[0])
        log_likelihood = float(
            np.sum(
                self.train_y * np.log(self.train_prob)
                + (1.0 - self.train_y) * np.log(1.0 - self.train_prob)
            )
        )
        parameter_count = int(len(coefficients) + 1)
        active_count = int(np.sum(np.abs(coefficients) > 1e-8) + (abs(intercept) > 1e-8))
        return CandidateFitStats(
            model_name=self.model_name,
            display_name=self.display_name,
            parameter_count=parameter_count,
            active_parameter_count=active_count,
            n_features=len(self._all_features()),
            train_log_likelihood=log_likelihood,
            train_deviance=float(-2.0 * log_likelihood),
            train_aic=float("nan"),
            train_bic=float("nan"),
            notes=f"hinge_features={len(self.hinge_features)}; knots_per_feature={self.knots_per_feature}; interaction_degree={self.interaction_degree}",
        )
