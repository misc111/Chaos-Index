# NBA Profit Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a profit-first NBA challenger tournament with denser penalized-GLM lambda search, broader nonlinear challenger support, and promotion-ready materialization for winning challengers.

**Architecture:** Extend the research and training layers in four slices: shared penalized-GLM search utilities, profit-first research ranking and promotion logic, trainable nonlinear challenger model wrappers, and end-to-end tests. Keep existing interfaces compatible where possible by adding richer metadata rather than replacing current fields outright.

**Tech Stack:** Python, pandas, scikit-learn, statsmodels, pydantic configs, SQLite-backed services, pytest

---

### Task 1: Add Shared Lambda Search Utilities

**Files:**
- Create: `src/training/lambda_search.py`
- Modify: `src/training/tune.py`
- Modify: `src/research/model_comparison.py`
- Test: `tests/test_model_selection.py`

- [ ] **Step 1: Write the failing test**

```python
from src.training.lambda_search import default_lambda_grid, lambda_to_c, penalized_glm_search_grid


def test_penalized_glm_search_grid_exposes_lambda_and_c_for_elastic_net() -> None:
    rows = penalized_glm_search_grid("glm_elastic_net")

    assert rows
    assert all("lambda" in row for row in rows)
    assert all("c" in row for row in rows)
    assert any("l1_ratio" in row for row in rows)
    assert rows[0]["lambda"] > 0
    assert rows[0]["c"] == lambda_to_c(rows[0]["lambda"])


def test_default_lambda_grid_is_dense_and_sorted_descending() -> None:
    grid = default_lambda_grid("glm_ridge")

    assert len(grid) >= 9
    assert grid == sorted(grid, reverse=True)
    assert min(grid) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_selection.py -k "lambda_grid or penalized_glm_search_grid" -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `src.training.lambda_search`

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/lambda_search.py
from __future__ import annotations

from math import isfinite


def lambda_to_c(value: float) -> float:
    lam = float(value)
    if not isfinite(lam) or lam <= 0:
        raise ValueError("lambda must be positive and finite")
    return 1.0 / lam


def default_lambda_grid(model_name: str) -> list[float]:
    token = str(model_name or "").strip()
    if token == "glm_ridge":
        return [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    if token == "glm_lasso":
        return [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
    if token == "glm_elastic_net":
        return [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
    raise ValueError(f"Unsupported penalized GLM '{model_name}'")


def penalized_glm_search_grid(model_name: str) -> list[dict[str, float]]:
    lambdas = default_lambda_grid(model_name)
    if model_name == "glm_elastic_net":
        return [
            {"lambda": lam, "c": lambda_to_c(lam), "l1_ratio": ratio}
            for lam in lambdas
            for ratio in [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
        ]
    return [{"lambda": lam, "c": lambda_to_c(lam)} for lam in lambdas]
```

- [ ] **Step 4: Thread the shared utility into the tuning paths**

```python
# src/training/tune.py
from src.training.lambda_search import default_lambda_grid, lambda_to_c


def _default_c_grid(model_name: str) -> list[float]:
    return [lambda_to_c(value) for value in default_lambda_grid(model_name)]


def _default_tune_result(model_name: str) -> dict:
    config = penalized_glm_config(model_name)
    default_lambda = 1.0 / float(config.default_c)
    best_params = {"lambda": default_lambda, "c": float(config.default_c)}
    return {
        "best_lambda": default_lambda,
        "best_c": float(config.default_c),
        "best_l1_ratio": None if config.default_l1_ratio is None else float(config.default_l1_ratio),
        "best_params": best_params,
        "results": [],
        "fold_metrics": [],
    }
```

```python
# src/research/model_comparison.py
from src.training.lambda_search import penalized_glm_search_grid

CandidateSpec(
    model_name="glm_ridge",
    display_name="GLM Ridge",
    param_grid=penalized_glm_search_grid("glm_ridge"),
    builder=lambda fs, params: PenalizedLogitCandidate(
        model_name="glm_ridge",
        display_name="GLM Ridge",
        features=fs.screened_features,
        penalty="l2",
        c=float(params["c"]),
        solver="lbfgs",
    ),
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_model_selection.py -k "lambda_grid or penalized_glm_search_grid" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_model_selection.py src/training/lambda_search.py src/training/tune.py src/research/model_comparison.py
git commit -m "feat: add dense lambda search utilities"
```

### Task 2: Make Research Ranking Profit-First

**Files:**
- Modify: `src/services/research_backtest.py`
- Modify: `src/services/research_desk.py`
- Test: `tests/test_research_backtest.py`
- Test: `tests/test_research_desk_service.py`

- [ ] **Step 1: Write the failing test**

```python
from src.services.research_backtest import _choose_best_candidate, _promotion_summary


def test_choose_best_candidate_prefers_profit_over_better_log_loss() -> None:
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "glm_ridge",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5120.0,
                "mean_net_profit": 120.0,
                "mean_roi": 0.021,
                "profitable_folds": 2,
                "mean_max_drawdown": 300.0,
                "mean_log_loss": 0.59,
                "mean_brier": 0.20,
            },
            {
                "model_name": "gam_spline",
                "strategy": "riskAdjusted",
                "mean_ending_bankroll": 5400.0,
                "mean_net_profit": 400.0,
                "mean_roi": 0.031,
                "profitable_folds": 3,
                "mean_max_drawdown": 325.0,
                "mean_log_loss": 0.64,
                "mean_brier": 0.22,
            },
        ]
    )

    assert _choose_best_candidate(scorecard, baseline_model="glm_ridge") == "gam_spline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_research_backtest.py -k "prefers_profit_over_better_log_loss" -v`
Expected: FAIL because the current ranking still leans on `mean_log_loss`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/research_backtest.py
def _choose_best_candidate(scorecard: pd.DataFrame, *, baseline_model: str) -> str:
    candidates = scorecard[scorecard["model_name"] != baseline_model].copy()
    if candidates.empty:
        return baseline_model
    ranked = candidates.sort_values(
        [
            "mean_ending_bankroll",
            "mean_net_profit",
            "mean_roi",
            "profitable_folds",
            "mean_max_drawdown",
            "mean_log_loss",
            "mean_brier",
        ],
        ascending=[False, False, False, False, True, True, True],
    )
    return str(ranked.iloc[0]["model_name"])
```

```python
# src/services/research_backtest.py
checks = {
    "mean_ending_bankroll": float(chosen["mean_ending_bankroll"]) > float(baseline["mean_ending_bankroll"]),
    "mean_net_profit": float(chosen["mean_net_profit"]) > float(baseline["mean_net_profit"]),
    "median_roi": float(chosen["median_roi"]) >= float(baseline["median_roi"]),
    "outer_fold_profit_wins": int(chosen["profitable_folds"]) >= int(baseline["profitable_folds"]),
    "drawdown_guardrail": float(chosen["mean_max_drawdown"]) <= float(baseline["mean_max_drawdown"]) + 250.0,
    "ece_guardrail": float(chosen["mean_ece"]) <= float(baseline["mean_ece"]) + 0.02,
    "integrity_checks": bool(chosen["all_integrity_checks"]),
}
```

- [ ] **Step 4: Relax the research-desk promotion gate so profit wins lead and metrics guard**

```python
# src/services/research_desk.py
MATERIALIZABLE_MODEL_NAMES = {
    "ensemble",
    "glm_elastic_net",
    "glm_lasso",
    "glm_ridge",
    "glm_vanilla",
    "gam_spline",
    "mars_hinge",
    "glmm_logit",
    "dglm_margin",
    "dynamic_rating",
    "bayes_bt_state_space",
}

gates = {
    "research_backtest_eligible": bool(promotion.get("eligible")),
    "materializable_candidate": _materializable_candidate(candidate_model_name),
    "beats_incumbent_bankroll": float(best_row.get("mean_ending_bankroll") or 0.0) > float(baseline_row.get("mean_ending_bankroll") or 0.0),
    "beats_incumbent_profit": float(best_row.get("mean_net_profit") or 0.0) > float(baseline_row.get("mean_net_profit") or 0.0),
    "max_drawdown_limit": float(best_row.get("mean_max_drawdown") or 0.0) <= max_drawdown_limit,
    "minimum_bet_count": int(best_row.get("bet_count") or 0) >= min_bet_count,
    "minimum_profitable_folds": profitable_folds >= min_profitable_folds,
    "calibration_guardrail": float(best_row.get("mean_ece") or 0.0) <= float(baseline_row.get("mean_ece") or 0.0) + max_ece_delta,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_research_backtest.py tests/test_research_desk_service.py -v`
Expected: PASS with the new profit-first ranking and relaxed-but-explicit guardrails

- [ ] **Step 6: Commit**

```bash
git add tests/test_research_backtest.py tests/test_research_desk_service.py src/services/research_backtest.py src/services/research_desk.py
git commit -m "feat: rank research challengers by profit first"
```

### Task 3: Add Trainable Nonlinear Challenger Models

**Files:**
- Create: `src/models/challenger_prob.py`
- Modify: `src/registry/models.py`
- Modify: `src/training/fit_runner.py`
- Modify: `src/training/predict_runner.py`
- Modify: `web/lib/predictions-report.ts`
- Test: `tests/test_model_selection.py`

- [ ] **Step 1: Write the failing test**

```python
from src.training.model_catalog import normalize_selected_models


def test_normalize_selected_models_accepts_trainable_challengers() -> None:
    selected = normalize_selected_models(["gam_spline", "mars_hinge", "glmm_logit", "dglm_margin", "glm_vanilla"])
    assert selected == ["gam_spline", "mars_hinge", "glmm_logit", "dglm_margin", "glm_vanilla"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_selection.py -k "trainable_challengers" -v`
Expected: FAIL because the registry does not currently include these challenger names

- [ ] **Step 3: Write minimal wrapper models**

```python
# src/models/challenger_prob.py
from __future__ import annotations

import pandas as pd

from src.models.base import BaseProbModel
from src.research.candidate_models import (
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    MARSHingeCandidate,
    VanillaGLMBinomialCandidate,
)


class VanillaGLMModel(BaseProbModel):
    model_name = "glm_vanilla"

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_col: str = "home_win") -> None:
        self.feature_columns = list(feature_columns)
        self.model = VanillaGLMBinomialCandidate(features=self.feature_columns)
        self.model.fit(df, target_col=target_col)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)
```

```python
# same file
class GAMSplineModel(BaseProbModel):
    model_name = "gam_spline"


class MARSHingeModel(BaseProbModel):
    model_name = "mars_hinge"


class GLMMLogitModel(BaseProbModel):
    model_name = "glmm_logit"


class DGLMMarginModel(BaseProbModel):
    model_name = "dglm_margin"
```

- [ ] **Step 4: Register and fit the wrappers**

```python
# src/registry/models.py
ModelRegistryEntry(key="glm_vanilla", display_label="Vanilla GLM", short_label="Vanilla GLM", family="linear", prediction_report_rank=14),
ModelRegistryEntry(key="gam_spline", display_label="GAM Spline", short_label="GAM", family="nonlinear", prediction_report_rank=15),
ModelRegistryEntry(key="mars_hinge", display_label="MARS Hinge", short_label="MARS", family="nonlinear", prediction_report_rank=16),
ModelRegistryEntry(key="glmm_logit", display_label="GLMM Logit", short_label="GLMM", family="nonlinear", prediction_report_rank=17),
ModelRegistryEntry(key="dglm_margin", display_label="DGLM Margin", short_label="DGLM", family="nonlinear", prediction_report_rank=18),
```

```python
# src/training/fit_runner.py
from src.models.challenger_prob import DGLMMarginModel, GAMSplineModel, GLMMLogitModel, MARSHingeModel, VanillaGLMModel

if "glm_vanilla" in selected:
    model = VanillaGLMModel()
    model.fit(train_df, resolve_model_feature_columns(...))
    models[model.model_name] = model

if "gam_spline" in selected:
    model = GAMSplineModel()
    model.fit(train_df, resolve_model_feature_columns(...))
    models[model.model_name] = model
```

- [ ] **Step 5: Update user-facing labels**

```typescript
// web/lib/predictions-report.ts
gam_spline: "GAM Spline",
mars_hinge: "MARS Hinge",
glmm_logit: "GLMM Logit",
dglm_margin: "DGLM Margin",
glm_vanilla: "Vanilla GLM",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_model_selection.py -v`
Expected: PASS with the new trainable challenger names

- [ ] **Step 7: Commit**

```bash
git add tests/test_model_selection.py src/models/challenger_prob.py src/registry/models.py src/training/fit_runner.py src/training/predict_runner.py web/lib/predictions-report.ts
git commit -m "feat: make nonlinear challengers trainable"
```

### Task 4: Broaden The NBA Research Desk Defaults

**Files:**
- Modify: `configs/research_briefs/nba/default.yaml`
- Test: `tests/test_research_desk.py`

- [ ] **Step 1: Write the failing test**

```python
def test_default_research_brief_includes_broad_challenger_field():
    cfg = load_config("configs/nba.yaml")
    briefs = load_structured_research_briefs(cfg)
    models = briefs[0].candidate_models

    assert "glm_elastic_net" in models
    assert "gam_spline" in models
    assert "mars_hinge" in models
    assert "glmm_logit" in models
    assert "dglm_margin" in models
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_research_desk.py -k "broad_challenger_field" -v`
Expected: FAIL because the default brief only lists the current GLM family

- [ ] **Step 3: Write minimal implementation**

```yaml
# configs/research_briefs/nba/default.yaml
candidate_models:
  - glm_elastic_net
  - glm_ridge
  - glm_lasso
  - glm_vanilla
  - gam_spline
  - mars_hinge
  - glmm_logit
  - dglm_margin
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_research_desk.py -k "broad_challenger_field" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_research_desk.py configs/research_briefs/nba/default.yaml
git commit -m "feat: broaden NBA research desk challenger defaults"
```

### Task 5: Full Verification

**Files:**
- Modify: `docs/generated/manifests.md` (only if registry generation changes it)
- Modify: `configs/generated/model_manifest.json` (only if registry generation changes it)

- [ ] **Step 1: Regenerate generated registry artifacts**

Run: `python3 -m src.registry.generate`
Expected: generated manifests reflect the newly trainable challenger models

- [ ] **Step 2: Run the targeted Python test suite**

Run: `pytest tests/test_model_selection.py tests/test_research_backtest.py tests/test_research_desk.py tests/test_research_desk_service.py tests/test_registry_contracts.py -v`
Expected: PASS

- [ ] **Step 3: Run the relevant web test if the TS label changes require it**

Run: `cd web && npm test -- --runInBand lib/server/services/betting-driver.test.ts`
Expected: PASS or, if the project does not expose this runner, document that the TS unit test command is unavailable and rely on typecheck instead

- [ ] **Step 4: Run type and registry verification**

Run: `make typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src tests configs web/lib docs/generated configs/generated
git commit -m "feat: add profit-first NBA challenger tournament support"
```
