"""Compatibility shim only; implementations live in extensions/ or experimental/."""

from src.models.experimental.mars_hinge import MARSHingeModel
from src.models.extensions.glm_variants import (
    DGLMMarginModel,
    GAMSplineModel,
    GLMMLogitModel,
    VanillaGLMModel,
)

__all__ = [
    "DGLMMarginModel",
    "GAMSplineModel",
    "GLMMLogitModel",
    "MARSHingeModel",
    "VanillaGLMModel",
]
