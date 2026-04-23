# Experimental Challengers

This package contains non-CAS challenger models that remain useful for
benchmarking or opt-in research, but are not part of the default MLB theory
lane.

Included here:

- `gbdt.py`
- `rf.py`
- `nn.py`
- `bayes_state_space_bt.py`
- `bayes_state_space_goals.py`
- `mars_hinge.py`
- `two_stage.py`

Usage rules:

- Import challenger models from `src.models.experimental.*`.
- Keep CAS-governed GLM, penalized-GLM, lasso-credibility, and theory-backed
  extensions out of this package unless their current implementation is a
  proxy that is not directly justified by the governing monographs.
- Legacy top-level challenger modules exist only as compatibility shims for
  older callers outside the primary MLB training lane.
