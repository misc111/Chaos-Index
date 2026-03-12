"""Model feature guardrail loading and filtering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.registry.models import legacy_model_keys


MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE = "configs/model_feature_guardrails_{league}.yaml"
MODEL_GUARDRAIL_FALLBACKS = {
    str(model_name): tuple(str(item) for item in values)
    for model_name, values in legacy_model_keys().items()
}


def default_guardrails_path_template(path_template: str) -> str:
    """Normalize a feature-map template into the guardrails companion template."""

    rendered = str(path_template or "")
    if "model_feature_map_" in rendered:
        return rendered.replace("model_feature_map_", "model_feature_guardrails_")
    return MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE


def resolve_model_feature_guardrails_path(path_template: str, league: str) -> Path:
    """Resolve the guardrails config path for a given league."""

    league_token = str(league or "unknown").strip().lower()
    rendered = str(path_template).replace("{league}", league_token)
    return Path(rendered)


def _load_guardrails(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _normalize_feature_map(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for feature_name, payload in raw.items():
        name = str(feature_name).strip()
        if not name:
            continue
        out[name] = dict(payload) if isinstance(payload, dict) else {"note": str(payload)}
    return out


def load_model_feature_guardrails(
    league: str,
    *,
    path_template: str = MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE,
) -> dict[str, dict[str, Any]]:
    """Load per-model blocked and watchlist feature metadata for a league."""

    path = resolve_model_feature_guardrails_path(path_template, league=league)
    raw = _load_guardrails(path)
    models = raw.get("models", {})
    if not isinstance(models, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for model_name, payload in models.items():
        if not isinstance(payload, dict):
            continue
        out[str(model_name)] = {
            "blocked_features": _normalize_feature_map(payload.get("blocked_features", {})),
            "watchlist_features": _normalize_feature_map(payload.get("watchlist_features", {})),
            "watchlist_pairs": list(payload.get("watchlist_pairs", []))
            if isinstance(payload.get("watchlist_pairs", []), list)
            else [],
        }
    return out


def blocked_features_for_model(
    league: str,
    model_name: str,
    *,
    path_template: str = MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE,
) -> set[str]:
    """Return the blocked feature set for a model, honoring legacy fallback keys."""

    guardrails = load_model_feature_guardrails(league, path_template=path_template)
    candidate_keys = [str(model_name)] + list(MODEL_GUARDRAIL_FALLBACKS.get(str(model_name), ()))
    for key in candidate_keys:
        payload = guardrails.get(key, {})
        blocked = payload.get("blocked_features", {})
        if isinstance(blocked, dict) and blocked:
            return set(blocked.keys())
    return set()


def apply_model_feature_guardrails(
    features: list[str],
    *,
    league: str,
    model_name: str,
    path_template: str = MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE,
) -> tuple[list[str], list[str]]:
    """Partition candidate features into approved and blocked lists."""

    blocked = blocked_features_for_model(league, model_name, path_template=path_template)
    approved: list[str] = []
    blocked_hits: list[str] = []
    seen: set[str] = set()

    for raw_feature in features:
        feature = str(raw_feature).strip()
        if not feature or feature in seen:
            continue
        seen.add(feature)
        if feature in blocked:
            blocked_hits.append(feature)
            continue
        approved.append(feature)
    return approved, blocked_hits


def find_model_feature_guardrail_conflicts(
    features: list[str],
    *,
    league: str,
    model_name: str,
    path_template: str = MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE,
) -> list[str]:
    """Return only the blocked feature hits for a proposed feature set."""

    _, blocked_hits = apply_model_feature_guardrails(
        features,
        league=league,
        model_name=model_name,
        path_template=path_template,
    )
    return blocked_hits
