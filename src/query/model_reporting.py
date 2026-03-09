"""Shared model-name normalization and ordering for query responses."""

from __future__ import annotations

from typing import Any

from src.training.model_catalog import MODEL_ALIASES, MODEL_REPORT_ORDER


MODEL_TRUST_NOTES = {
    "ensemble": "All models combined. Best default pick. Can share the same blind spot.",
    "elo_baseline": "Standard sports betting baseline based on past wins/losses. Good long-run read. Slow on sudden changes.",
    "glm_ridge": "Statistical model that uses a checklist. Usually steady. Weird matchups can slip through.",
    "glm_lasso": "Statistical model that can drop whole inputs instead of just shrinking them. Good for leaner signal sets. Can miss smaller shared effects.",
    "glm_elastic_net": "Statistical model that mixes ridge-style shrinkage with lasso-style pruning. Good when signals travel in clusters. Can still mute small but real effects.",
    "dynamic_rating": "Hot/cold meter. Good for momentum. Can overreact to short streaks.",
    "gbdt": "Machine learning model that finds hidden combos. Sometimes too confident.",
    "rf": "Machine learning model that blends many different predictions from random slices of past games. Good at smoothing out flukes. Can be too cautious on close matchups.",
    "two_stage": "Machine learning model with two steps: first predicts game type (fast/slow, close/lopsided), then predicts winner. Good when style matchups matter. If step 1 is wrong, final pick can be wrong.",
    "goals_poisson": "Score-based model. Good for normal scoring games. Messy games hurt it.",
    "simulation_first": "Runs the matchup thousands of times using set assumptions (team strength, pace, and scoring). Good for seeing different paths. If those assumptions are off, this number can be off.",
    "bayes_bt_state_space": "Tracks team strength after every game and gives a range, not just one number. Good for spotting rising/falling teams with uncertainty shown. Can move fast after injuries, trades, or short weird stretches.",
    "bayes_goals": "Scoring strength + confidence meter. Good trend read. Can lag sudden lineup changes.",
    "nn_mlp": "Machine learning model that finds subtle patterns. Hardest to explain.",
}


def canonical_model_name(model_name: Any) -> str:
    token = str(model_name or "").strip()
    return MODEL_ALIASES.get(token, token)


def canonicalize_model_probabilities(per_model: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_name, raw_value in per_model.items():
        model_name = canonical_model_name(raw_name)
        if not model_name:
            continue
        try:
            out[model_name] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return out


def ordered_model_names(model_names: set[str]) -> list[str]:
    canonical_names = {canonical_model_name(name) for name in model_names if canonical_model_name(name)}
    ordered = [name for name in MODEL_REPORT_ORDER if name in canonical_names]
    seen = set(ordered)
    ordered.extend(sorted(name for name in canonical_names if name not in seen))
    return ordered
