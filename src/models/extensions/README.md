# Theory-Compatible Extensions

This package contains GLM-adjacent extension implementations that are useful
challengers but are not the default MLB theory-core production lane.

Included here:

- `glm_variants.py`: GAM, GLMM, and DGLM wrappers.
- `glm_goals.py`: score/count GLM-style goal/run model bridge.
- `ensemble_stack.py` and `ensemble_weighted.py`: opt-in ensemble helpers.

Usage rules:

- Import extension models from `src.models.extensions.*`.
- Keep top-level `src.models.*` files as compatibility shims only, except for
  shared contracts such as `src.models.base`.
- Promotion artifacts must label these rows as `theory-compatible extension`.
