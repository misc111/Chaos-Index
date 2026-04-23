"""Compatibility shim only; stacking ensemble code lives in src.models.extensions."""

from src.models.extensions.ensemble_stack import StackingEnsemble

__all__ = ["StackingEnsemble"]
