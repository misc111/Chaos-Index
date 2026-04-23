"""Compatibility shim only; score/count GLM code lives in src.models.extensions."""

from src.models.extensions.glm_goals import GoalsModelOutput, GoalsPoissonModel

__all__ = ["GoalsModelOutput", "GoalsPoissonModel"]
