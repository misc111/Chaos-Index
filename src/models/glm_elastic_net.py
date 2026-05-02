"""Compatibility shim for the core elastic-net GLM.

New code should import `GLMElasticNetModel` from `src.models.core`; this module
is retained only for older call sites and external references.
"""

from src.models.core.glm_penalized import GLMElasticNetModel

__all__ = ["GLMElasticNetModel"]
