"""Canonical import surface for CAS-governed core model implementations.

The core namespace is reserved for monograph-backed GLM-family models that are
eligible for the MLB default theory lane. Top-level `src.models.*` modules may
re-export these names for backwards compatibility, but new runtime code should
import from this package or its concrete implementation modules.
"""

from src.models.core.glm_penalized import (
    ACTIVE_COEF_TOLERANCE,
    GLMElasticNetModel,
    GLMLassoModel,
    GLMRidgeModel,
    PenalizedGLMConfig,
    PenalizedGLMModel,
    build_penalized_glm,
    penalized_glm_config,
)
from src.models.core.glm_vanilla import VanillaGLMModel
from src.models.core.lasso_credibility import (
    LassoCredibilityModel,
    default_lasso_credibility_label,
    normalize_lasso_credibility_kind,
)

__all__ = [
    "ACTIVE_COEF_TOLERANCE",
    "GLMElasticNetModel",
    "GLMLassoModel",
    "GLMRidgeModel",
    "LassoCredibilityModel",
    "PenalizedGLMConfig",
    "PenalizedGLMModel",
    "VanillaGLMModel",
    "build_penalized_glm",
    "default_lasso_credibility_label",
    "normalize_lasso_credibility_kind",
    "penalized_glm_config",
]
