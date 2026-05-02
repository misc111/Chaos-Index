"""Core-supported vanilla binomial/logit GLM wrapper.

This module gives the unpenalized GLM a first-class home in the explicit core
namespace. The implementation deliberately delegates the statistical fit to the
existing research candidate so the refactor changes import boundaries without
changing model math, artifact contracts, or prediction behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.models.base import BaseProbModel
from src.research.candidate_models import VanillaGLMBinomialCandidate


def _unique_features(feature_columns: Sequence[str]) -> list[str]:
    """Return stable, non-empty feature names without duplicates."""

    ordered = [str(column) for column in feature_columns if str(column).strip()]
    return list(dict.fromkeys(ordered))


class VanillaGLMModel(BaseProbModel):
    """Training-contract adapter for the core vanilla GLM candidate."""

    model_name = "glm_vanilla"

    def __init__(self) -> None:
        super().__init__()
        self._candidate: VanillaGLMBinomialCandidate | None = None

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        """Fit the vanilla GLM on a pregame feature frame."""

        self.feature_columns = _unique_features(feature_columns)
        self._candidate = VanillaGLMBinomialCandidate(features=self.feature_columns)
        self._candidate.fit(df, target_col=target_col)

    def predict_proba(self, df: pd.DataFrame):
        """Return clipped home-win probabilities for the fitted GLM."""

        if self._candidate is None:
            raise RuntimeError(f"{self.model_name} has not been fit")
        return self._candidate.predict_proba(df)
