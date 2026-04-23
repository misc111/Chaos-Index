"""Compatibility shim only; weighted ensemble code lives in src.models.extensions."""

from src.models.extensions.ensemble_weighted import compute_weights, spread_stats, weighted_ensemble

__all__ = ["compute_weights", "spread_stats", "weighted_ensemble"]
