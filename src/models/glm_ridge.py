"""Compatibility shim for the core ridge GLM.

New code should import `GLMRidgeModel` from `src.models.core`; this module is
retained only for older call sites and external references.
"""

from src.models.core.glm_penalized import GLMRidgeModel

__all__ = ["GLMRidgeModel"]
