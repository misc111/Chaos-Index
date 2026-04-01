from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.models.base import BaseProbModel
from src.research.candidate_models import (
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    MARSHingeCandidate,
    VanillaGLMBinomialCandidate,
)


def _unique_features(feature_columns: Sequence[str], *, limit: int | None = None) -> list[str]:
    ordered = [str(column) for column in feature_columns if str(column).strip()]
    deduped = list(dict.fromkeys(ordered))
    if limit is None:
        return deduped
    return deduped[: max(1, int(limit))]


def _split_linear_and_nonlinear(feature_columns: Sequence[str]) -> tuple[list[str], list[str]]:
    ordered = _unique_features(feature_columns)
    linear = ordered[: min(6, len(ordered))]
    nonlinear = ordered[len(linear) : len(linear) + 4]
    if not nonlinear:
        nonlinear = linear[: min(3, len(linear))]
    return linear, nonlinear


class _WrappedCandidateModel(BaseProbModel):
    candidate = None

    def __init__(self) -> None:
        super().__init__()
        self._candidate = None

    def predict_proba(self, df: pd.DataFrame):
        if self._candidate is None:
            raise RuntimeError(f"{self.model_name} has not been fit")
        return self._candidate.predict_proba(df)


class VanillaGLMModel(_WrappedCandidateModel):
    model_name = "glm_vanilla"

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        self.feature_columns = _unique_features(feature_columns)
        self._candidate = VanillaGLMBinomialCandidate(features=self.feature_columns)
        self._candidate.fit(df, target_col=target_col)


class GAMSplineModel(_WrappedCandidateModel):
    model_name = "gam_spline"

    def __init__(self, *, c: float = 0.5, n_knots: int = 5) -> None:
        super().__init__()
        self.c = float(c)
        self.n_knots = int(n_knots)

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        linear, spline = _split_linear_and_nonlinear(feature_columns)
        self.feature_columns = list(dict.fromkeys(linear + spline))
        self._candidate = GAMSplineCandidate(
            linear_features=linear,
            spline_features=spline,
            c=self.c,
            n_knots=self.n_knots,
        )
        self._candidate.fit(df, target_col=target_col)


class MARSHingeModel(_WrappedCandidateModel):
    model_name = "mars_hinge"

    def __init__(self, *, c: float = 0.25, knots_per_feature: int = 3, interaction_degree: int = 1) -> None:
        super().__init__()
        self.c = float(c)
        self.knots_per_feature = int(knots_per_feature)
        self.interaction_degree = int(interaction_degree)

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        linear, hinge = _split_linear_and_nonlinear(feature_columns)
        self.feature_columns = list(dict.fromkeys(linear + hinge))
        self._candidate = MARSHingeCandidate(
            linear_features=linear,
            hinge_features=hinge,
            knots_per_feature=self.knots_per_feature,
            interaction_degree=self.interaction_degree,
            c=self.c,
        )
        self._candidate.fit(df, target_col=target_col)


class GLMMLogitModel(_WrappedCandidateModel):
    model_name = "glmm_logit"

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        self.feature_columns = _unique_features(feature_columns, limit=8)
        self._candidate = GLMMLogitCandidate(fixed_features=self.feature_columns)
        self._candidate.fit(df, target_col=target_col)


class DGLMMarginModel(_WrappedCandidateModel):
    model_name = "dglm_margin"

    def __init__(self, *, iterations: int = 2) -> None:
        super().__init__()
        self.iterations = int(iterations)

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        self.feature_columns = _unique_features(feature_columns, limit=10)
        self._candidate = DGLMMarginCandidate(features=self.feature_columns, iterations=self.iterations)
        self._candidate.fit(df, target_col=target_col)
