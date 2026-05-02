"""Compatibility shim only; implementations live in core/extensions/experimental."""

from src.models.experimental.mars_hinge import MARSHingeModel
from src.models.core.glm_vanilla import VanillaGLMModel
from src.models.extensions.glm_variants import (
    DGLMMarginModel,
    GAMSplineModel,
    GLMMLogitModel,
)

__all__ = [
    "DGLMMarginModel",
    "GAMSplineModel",
    "GLMMLogitModel",
    "MARSHingeModel",
    "VanillaGLMModel",
]
