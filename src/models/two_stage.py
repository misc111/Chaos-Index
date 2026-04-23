"""Compatibility shim only; two-stage proxy code lives in src.models.experimental."""

from src.models.experimental.two_stage import TwoStageModel

__all__ = ["TwoStageModel"]
