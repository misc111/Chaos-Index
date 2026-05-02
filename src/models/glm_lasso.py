"""Compatibility shim for the core lasso GLM.

New code should import `GLMLassoModel` from `src.models.core`; this module is
retained only for older call sites and external references.
"""

from src.models.core.glm_penalized import GLMLassoModel

__all__ = ["GLMLassoModel"]
