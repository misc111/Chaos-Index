"""Theory-compatible extension model implementations.

These models are traceable to GLM-family extension ideas in the governing CAS
monographs, but they are not part of the default theory-core production lane.
"""

from src.models.extensions.ensemble_stack import StackingEnsemble
from src.models.extensions.ensemble_weighted import compute_weights, spread_stats, weighted_ensemble
from src.models.extensions.glm_goals import GoalsModelOutput, GoalsPoissonModel
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
    "GoalsModelOutput",
    "GoalsPoissonModel",
    "StackingEnsemble",
    "VanillaGLMModel",
    "compute_weights",
    "spread_stats",
    "weighted_ensemble",
]
