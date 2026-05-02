"""Model namespaces and compatibility shims.

CAS-governed core models live under ``src.models.core``. Opt-in
theory-compatible extensions live under ``src.models.extensions``. Experimental
challenger and proxy models live under ``src.models.experimental``. Top-level
model files remain compatibility shims for older callers, except for shared
contracts such as ``src.models.base``.
"""
