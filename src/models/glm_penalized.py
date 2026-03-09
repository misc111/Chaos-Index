from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseProbModel


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

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        train = df[df[target_col].notna()].copy()
        self.feature_columns = list(feature_columns)
        y = train[target_col].astype(int).to_numpy()
        x = self._prepare_x(train, fit=True)
        self.model.fit(x, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._prepare_x(df, fit=False)
        p = self.model.predict_proba(x)[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)

    def coef_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "coef": self.model.coef_[0],
            }
        ).sort_values("coef", key=lambda s: np.abs(s), ascending=False)


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
    if config.model_name == "glm_elastic_net":
        return GLMElasticNetModel(
            c=config.default_c if c is None else c,
            l1_ratio=config.default_l1_ratio if l1_ratio is None else l1_ratio,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported penalized GLM '{model_name}'")
