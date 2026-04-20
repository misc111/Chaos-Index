"""Canonical model metadata shared across training and the web layer."""

from __future__ import annotations

from src.registry.models import (
    baseline_model_names,
    cas_core_family_catalog,
    core_model_names,
    default_training_model_names,
    experimental_model_names,
    legacy_model_keys,
    model_aliases,
    penalized_core_model_names,
    prediction_report_order,
    planned_credibility_model_catalog,
    planned_credibility_model_names,
    trainable_model_names,
)

ALL_MODEL_NAMES = trainable_model_names()
CORE_MODEL_NAMES = core_model_names()
PENALIZED_CORE_MODEL_NAMES = penalized_core_model_names()
BASELINE_MODEL_NAMES = baseline_model_names()
EXPERIMENTAL_MODEL_NAMES = experimental_model_names()
PLANNED_CREDIBILITY_MODEL_NAMES = planned_credibility_model_names()
PLANNED_CREDIBILITY_MODEL_CATALOG = planned_credibility_model_catalog()
CAS_CORE_FAMILY_CATALOG = cas_core_family_catalog()
MODEL_ALIASES = model_aliases()
LEGACY_MODEL_KEYS = legacy_model_keys()
MODEL_REPORT_ORDER = prediction_report_order()
DEFAULT_MODEL_NAMES = default_training_model_names()


def normalize_selected_models(selected_models: list[str] | None) -> list[str]:
    """Normalize user-selected model tokens into canonical registered keys."""

    if not selected_models:
        return list(DEFAULT_MODEL_NAMES)

    out: list[str] = []
    bad: list[str] = []
    seen = set()
    for raw in selected_models:
        token = str(raw).strip().lower()
        if not token:
            continue
        if token in {"all", "*"}:
            return list(ALL_MODEL_NAMES)
        canonical = MODEL_ALIASES.get(token, token)
        if canonical not in ALL_MODEL_NAMES:
            bad.append(raw)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)

    if bad:
        raise ValueError(f"Unknown model names: {sorted(set(bad))}. Valid={ALL_MODEL_NAMES}")
    if not out:
        raise ValueError(f"No valid models selected. Valid={ALL_MODEL_NAMES}")
    return out
