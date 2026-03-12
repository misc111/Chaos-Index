"""Canonical model metadata shared across training and the web layer."""

from __future__ import annotations

from src.registry.models import (
    legacy_model_keys,
    model_aliases,
    prediction_report_order,
    trainable_model_names,
)

ALL_MODEL_NAMES = trainable_model_names()
MODEL_ALIASES = model_aliases()
LEGACY_MODEL_KEYS = legacy_model_keys()
MODEL_REPORT_ORDER = prediction_report_order()


def normalize_selected_models(selected_models: list[str] | None) -> list[str]:
    """Normalize user-selected model tokens into canonical registered keys."""

    if not selected_models:
        return list(ALL_MODEL_NAMES)

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
