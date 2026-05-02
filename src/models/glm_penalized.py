"""Compatibility shim for core penalized GLM implementations.

The canonical implementations live in `src.models.core.glm_penalized`. This
module exists only so older imports keep working while production code migrates
to the explicit core namespace.
"""

from src.models.core.glm_penalized import (
    ACTIVE_COEF_TOLERANCE,
    PENALIZED_GLM_CONFIGS,
    PENALIZED_GLM_MODEL_NAMES,
    GLMElasticNetModel,
    GLMLassoModel,
    GLMRidgeModel,
    PenalizedGLMConfig,
    PenalizedGLMModel,
    build_penalized_glm,
    penalized_glm_config,
)

__all__ = [
    "ACTIVE_COEF_TOLERANCE",
    "PENALIZED_GLM_CONFIGS",
    "PENALIZED_GLM_MODEL_NAMES",
    "GLMElasticNetModel",
    "GLMLassoModel",
    "GLMRidgeModel",
    "PenalizedGLMConfig",
    "PenalizedGLMModel",
    "build_penalized_glm",
    "penalized_glm_config",
]
