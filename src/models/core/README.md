# Core Models

This package owns the canonical CAS-governed model implementations:

- vanilla binomial/logit GLM
- ridge, lasso, and elastic-net penalized GLMs
- market-offset and prior-offset lasso-credibility families

Top-level `src.models.*` files are compatibility shims only. New runtime imports
should use `src.models.core.*` for these core families.
