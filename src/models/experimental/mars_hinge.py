"""Experimental hinge-basis proxy for the MARS challenger lane."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.models.base import BaseProbModel
from src.research.candidate_models import MARSHingeCandidate


def _unique_features(feature_columns: Sequence[str], *, limit: int | None = None) -> list[str]:
    ordered = [str(column) for column in feature_columns if str(column).strip()]
    deduped = list(dict.fromkeys(ordered))
    if limit is None:
        return deduped
    return deduped[: max(1, int(limit))]


def _split_linear_and_hinge(feature_columns: Sequence[str]) -> tuple[list[str], list[str]]:
    ordered = _unique_features(feature_columns)
    linear = ordered[: min(6, len(ordered))]
    hinge = ordered[len(linear) : len(linear) + 4]
    if not hinge:
        hinge = linear[: min(3, len(linear))]
    return linear, hinge


class MARSHingeModel(BaseProbModel):
    """Experimental fixed hinge-basis proxy, not full forward/pruned MARS."""

    model_name = "mars_hinge"

    def __init__(self, *, c: float = 0.25, knots_per_feature: int = 3, interaction_degree: int = 1) -> None:
        super().__init__()
        self.c = float(c)
        self.knots_per_feature = int(knots_per_feature)
        self.interaction_degree = int(interaction_degree)
        self._candidate = None

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        linear, hinge = _split_linear_and_hinge(feature_columns)
        self.feature_columns = list(dict.fromkeys(linear + hinge))
        self._candidate = MARSHingeCandidate(
            linear_features=linear,
            hinge_features=hinge,
            knots_per_feature=self.knots_per_feature,
            interaction_degree=self.interaction_degree,
            c=self.c,
        )
        self._candidate.fit(df, target_col=target_col)

    def predict_proba(self, df: pd.DataFrame):
        if self._candidate is None:
            raise RuntimeError("mars_hinge has not been fit")
        return self._candidate.predict_proba(df)
