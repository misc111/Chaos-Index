"""Canonical model metadata shared across training and the web layer."""

from __future__ import annotations

from src.common.manifests import load_model_manifest


MODEL_MANIFEST = load_model_manifest()
ALL_MODEL_NAMES = list(MODEL_MANIFEST["trainable_models"])
MODEL_ALIASES = dict(MODEL_MANIFEST["aliases"])
LEGACY_MODEL_KEYS = {
    str(model_name): tuple(str(item) for item in values)
    for model_name, values in dict(MODEL_MANIFEST.get("legacy_model_keys", {})).items()
}
MODEL_REPORT_ORDER = list(MODEL_MANIFEST["prediction_report_order"])


def normalize_selected_models(selected_models: list[str] | None) -> list[str]:
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
