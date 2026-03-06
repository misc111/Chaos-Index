from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE = "configs/model_feature_guardrails_{league}.yaml"


def default_guardrails_path_template(path_template: str) -> str:
    rendered = str(path_template or "")
    if "model_feature_map_" in rendered:
        return rendered.replace("model_feature_map_", "model_feature_guardrails_")
    return MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE


def resolve_model_feature_guardrails_path(path_template: str, league: str) -> Path:
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
    guardrails = load_model_feature_guardrails(league, path_template=path_template)
    payload = guardrails.get(str(model_name), {})
    blocked = payload.get("blocked_features", {})
    if not isinstance(blocked, dict):
        return set()
    return set(blocked.keys())


def apply_model_feature_guardrails(
    features: list[str],
    *,
    league: str,
    model_name: str,
    path_template: str = MODEL_FEATURE_GUARDRAILS_PATH_TEMPLATE,
) -> tuple[list[str], list[str]]:
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
    _, blocked_hits = apply_model_feature_guardrails(
        features,
        league=league,
        model_name=model_name,
        path_template=path_template,
    )
    return blocked_hits
