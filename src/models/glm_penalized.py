from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseProbModel

ACTIVE_COEF_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class PenalizedGLMConfig:
    model_name: str
    penalty: str
    solver: str
    default_c: float = 1.0
    default_l1_ratio: float | None = None
    max_iter: int = 4000


PENALIZED_GLM_CONFIGS: dict[str, PenalizedGLMConfig] = {
    "glm_ridge": PenalizedGLMConfig(
        model_name="glm_ridge",
        penalty="l2",
        solver="lbfgs",
        default_c=1.0,
    ),
    "glm_lasso": PenalizedGLMConfig(
        model_name="glm_lasso",
        penalty="l1",
        solver="saga",
        default_c=1.0,
    ),
    "glm_elastic_net": PenalizedGLMConfig(
        model_name="glm_elastic_net",
        penalty="elasticnet",
        solver="saga",
        default_c=1.0,
        default_l1_ratio=0.5,
    ),
}
PENALIZED_GLM_MODEL_NAMES = tuple(PENALIZED_GLM_CONFIGS.keys())


def penalized_glm_config(model_name: str) -> PenalizedGLMConfig:
    token = str(model_name or "").strip()
    if token not in PENALIZED_GLM_CONFIGS:
        raise ValueError(f"Unsupported penalized GLM '{model_name}'. Valid={list(PENALIZED_GLM_CONFIGS)}")
    return PENALIZED_GLM_CONFIGS[token]


class PenalizedGLMModel(BaseProbModel):
    model_name = "glm_penalized"

    def __init__(
        self,
        *,
        model_name: str,
        c: float | None = None,
        l1_ratio: float | None = None,
        random_state: int = 42,
    ):
        super().__init__()
        config = penalized_glm_config(model_name)
        self.model_name = config.model_name
        self.penalty = config.penalty
        self.solver = config.solver
        self.c = float(config.default_c if c is None else c)
        self.l1_ratio = config.default_l1_ratio if l1_ratio is None else float(l1_ratio)
        self.random_state = int(random_state)
        model_kwargs: dict[str, float | int | str] = {
            "penalty": self.penalty,
            "C": self.c,
            "max_iter": int(config.max_iter),
            "solver": self.solver,
            "random_state": self.random_state,
        }
        if self.penalty == "elasticnet":
            model_kwargs["l1_ratio"] = float(self.l1_ratio if self.l1_ratio is not None else 0.5)
        self.model = LogisticRegression(**model_kwargs)
        self.scaler = StandardScaler()
        self.feature_medians: dict[str, float] = {}
        self.fit_target_col = "home_win"
        self.fit_row_count = 0
        self.fit_positive_rate: float | None = None

    def _prepare_x(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        x = df[self.feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
        if fit:
            med = x.median(numeric_only=True).fillna(0.0)
            self.feature_medians = {str(k): float(v) for k, v in med.items()}
        fill = pd.Series(self.feature_medians)
        x = x.fillna(fill).fillna(0.0)
        if fit:
            return self.scaler.fit_transform(x.to_numpy(dtype=float))
        return self.scaler.transform(x.to_numpy(dtype=float))

    def penalty_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "penalty_family": self.model_name.removeprefix("glm_"),
            "penalty": self.penalty,
            "solver": self.solver,
            "lambda": float(1.0 / self.c),
            "c": float(self.c),
            "l1_ratio": None if self.l1_ratio is None else float(self.l1_ratio),
        }

    def _coef_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coef_scaled = np.asarray(self.model.coef_[0], dtype=float)
        scale = np.asarray(getattr(self.scaler, "scale_", np.ones(len(coef_scaled))), dtype=float)
        means = np.asarray(getattr(self.scaler, "mean_", np.zeros(len(coef_scaled))), dtype=float)
        coef_original = np.divide(
            coef_scaled,
            scale,
            out=np.full_like(coef_scaled, np.nan, dtype=float),
            where=np.isfinite(scale) & (np.abs(scale) > 0),
        )
        return coef_scaled, coef_original, scale, means

    def intercept_original(self) -> float | None:
        intercept_scaled = float(np.asarray(self.model.intercept_, dtype=float)[0])
        coef_scaled, _, scale, means = self._coef_arrays()
        mean_over_scale = np.divide(
            means,
            scale,
            out=np.zeros_like(means, dtype=float),
            where=np.isfinite(scale) & (np.abs(scale) > 0),
        )
        intercept_original = intercept_scaled - float(np.dot(coef_scaled, mean_over_scale))
        return intercept_original if np.isfinite(intercept_original) else None

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        train = df[df[target_col].notna()].copy()
        self.feature_columns = list(feature_columns)
        y = train[target_col].astype(int).to_numpy()
        x = self._prepare_x(train, fit=True)
        self.model.fit(x, y)
        self.fit_target_col = target_col
        self.fit_row_count = int(len(train))
        self.fit_positive_rate = float(y.mean()) if len(y) else None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._prepare_x(df, fit=False)
        p = self.model.predict_proba(x)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)

    def coef_frame(self, *, active_only: bool = False, top_n: int | None = None) -> pd.DataFrame:
        coef_scaled, coef_original, scale, means = self._coef_arrays()
        medians = (
            pd.Series(self.feature_medians, dtype=float)
            .reindex(self.feature_columns)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        frame = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "coef": coef_scaled,
                "coef_scaled": coef_scaled,
                "coef_original": coef_original,
                "abs_coef_scaled": np.abs(coef_scaled),
                "abs_coef_original": np.abs(coef_original),
                "sign": np.sign(coef_scaled).astype(int),
                "scale": scale,
                "feature_mean": means,
                "median_impute_value": medians,
                "active": np.abs(coef_scaled) > ACTIVE_COEF_TOLERANCE,
                "model_name": self.model_name,
                "penalty_family": self.model_name.removeprefix("glm_"),
                "lambda": float(1.0 / self.c),
                "c": float(self.c),
                "l1_ratio": None if self.l1_ratio is None else float(self.l1_ratio),
            }
        ).sort_values(["abs_coef_scaled", "feature"], ascending=[False, True], kind="mergesort")
        frame = frame.reset_index(drop=True)
        active_rank = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        if bool(frame["active"].any()):
            active_rank.loc[frame["active"]] = np.arange(1, int(frame["active"].sum()) + 1, dtype=int)
        frame["active_rank"] = active_rank
        if active_only:
            frame = frame[frame["active"]].reset_index(drop=True)
        if top_n is not None:
            frame = frame.head(max(int(top_n), 0)).reset_index(drop=True)
        return frame

    def active_features(self) -> list[str]:
        frame = self.coef_frame(active_only=True)
        return [str(feature) for feature in frame["feature"].tolist() if str(feature).strip()]

    def active_coefficient_summary(self, top_n: int = 8) -> list[dict[str, Any]]:
        frame = self.coef_frame(active_only=True, top_n=top_n)
        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "feature": str(row.feature),
                    "coef_scaled": float(row.coef_scaled),
                    "coef_original": float(row.coef_original) if np.isfinite(row.coef_original) else None,
                    "abs_coef_scaled": float(row.abs_coef_scaled),
                    "abs_coef_original": float(row.abs_coef_original) if np.isfinite(row.abs_coef_original) else None,
                    "sign": int(row.sign),
                    "active_rank": None if pd.isna(row.active_rank) else int(row.active_rank),
                }
            )
        return rows

    def coefficient_path_metadata(self, top_n: int = 8) -> dict[str, Any]:
        frame = self.coef_frame()
        return {
            "path_columns": [str(column) for column in frame.columns],
            "sort_key": "abs_coef_scaled_desc",
            "top_feature_names": [str(value) for value in frame.head(max(int(top_n), 0))["feature"].tolist()],
            "active_feature_names": self.active_features(),
            "parameterization": self.penalty_metadata(),
        }

    def fit_metadata_dict(self, top_n: int = 8) -> dict[str, Any]:
        intercept_scaled = float(np.asarray(self.model.intercept_, dtype=float)[0])
        active_features = self.active_features()
        return {
            **self.penalty_metadata(),
            "target_col": self.fit_target_col,
            "n_obs": int(self.fit_row_count),
            "feature_count": int(len(self.feature_columns)),
            "active_parameter_count": int(len(active_features)),
            "active_features": active_features,
            "active_coefficient_summary": self.active_coefficient_summary(top_n=top_n),
            "coefficient_columns": [str(column) for column in self.coef_frame().columns],
            "coefficient_path_metadata": self.coefficient_path_metadata(top_n=top_n),
            "intercept_scaled": intercept_scaled,
            "intercept_original": self.intercept_original(),
            "intercept_active": bool(abs(intercept_scaled) > ACTIVE_COEF_TOLERANCE),
            "positive_rate": self.fit_positive_rate,
        }


class GLMRidgeModel(PenalizedGLMModel):
    model_name = "glm_ridge"

    def __init__(self, c: float = 1.0, random_state: int = 42):
        super().__init__(model_name=self.model_name, c=c, random_state=random_state)


class GLMElasticNetModel(PenalizedGLMModel):
    model_name = "glm_elastic_net"

    def __init__(self, c: float = 1.0, l1_ratio: float = 0.5, random_state: int = 42):
        super().__init__(
            model_name=self.model_name,
            c=c,
            l1_ratio=l1_ratio,
            random_state=random_state,
        )


class GLMLassoModel(PenalizedGLMModel):
    model_name = "glm_lasso"

    def __init__(self, c: float = 1.0, random_state: int = 42):
        super().__init__(model_name=self.model_name, c=c, random_state=random_state)


def build_penalized_glm(
    model_name: str,
    *,
    c: float | None = None,
    l1_ratio: float | None = None,
    random_state: int = 42,
) -> PenalizedGLMModel:
    config = penalized_glm_config(model_name)
    if config.model_name == "glm_ridge":
        return GLMRidgeModel(c=config.default_c if c is None else c, random_state=random_state)
    if config.model_name == "glm_lasso":
        return GLMLassoModel(c=config.default_c if c is None else c, random_state=random_state)
    if config.model_name == "glm_elastic_net":
        return GLMElasticNetModel(
            c=config.default_c if c is None else c,
            l1_ratio=config.default_l1_ratio if l1_ratio is None else l1_ratio,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported penalized GLM '{model_name}'")
