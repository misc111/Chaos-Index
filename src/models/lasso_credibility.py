"""Compatibility shim for core lasso-credibility implementations.

The canonical Holmes/Casotto lasso-credibility implementation now lives in
`src.models.core.lasso_credibility`. This shim preserves legacy imports without
letting top-level `src.models` become an implementation lane again.
"""

from src.models.core.lasso_credibility import (
    ACTIVE_COEF_TOLERANCE,
    DEFAULT_COEFFICIENT_TOP_N,
    PROBABILITY_EPS,
    LassoCredibilityModel,
    default_lasso_credibility_label,
    normalize_lasso_credibility_kind,
)

__all__ = [
    "ACTIVE_COEF_TOLERANCE",
    "DEFAULT_COEFFICIENT_TOP_N",
    "PROBABILITY_EPS",
    "LassoCredibilityModel",
    "default_lasso_credibility_label",
    "normalize_lasso_credibility_kind",
]
