from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.common.time import utc_now_iso
from src.common.utils import stable_hash


@dataclass(frozen=True)
class FeaturePolicyResult:
    approved_feature_columns: list[str]
    registry_path: str
    mode: str
    added_features: list[str]
    removed_features: list[str]
    candidates_added: list[str]
    registry_updated: bool
    registry_created: bool


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_registry_path(path_template: str, league: str) -> Path:
    league_token = str(league or "unknown").strip().lower()
    rendered = str(path_template).replace("{league}", league_token)
    return Path(rendered)


def _load_registry(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_registry(path: Path, *, league: str, active_features: list[str], candidate_features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "league": str(league).upper(),
        "updated_at_utc": utc_now_iso(),
        "active_features": sorted(_ordered_unique(active_features)),
        "candidate_features": sorted(_ordered_unique(candidate_features)),
        "active_features_hash": stable_hash(sorted(_ordered_unique(active_features))),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def apply_feature_policy(
    feature_columns: list[str],
    *,
    league: str,
    mode: str,
    registry_path_template: str,
    approve_changes: bool,
) -> FeaturePolicyResult:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"production", "research"}:
        raise ValueError(f"Unsupported feature policy mode '{mode}'. Expected 'production' or 'research'.")

    current_ordered = _ordered_unique(feature_columns)
    current_set = set(current_ordered)
    registry_path = resolve_registry_path(registry_path_template, league=league)
    raw = _load_registry(registry_path)

    existing_active = _ordered_unique(list(raw.get("active_features", []))) if raw else []
    existing_candidates = _ordered_unique(list(raw.get("candidate_features", []))) if raw else []
    active_set = set(existing_active)
    candidate_set = set(existing_candidates)

    if not raw:
        if mode_norm == "production" and not approve_changes:
            raise RuntimeError(
                "Feature registry not found for production mode at "
                f"{registry_path}. Re-run with '--approve-feature-changes' to bootstrap the active feature contract."
            )
        _write_registry(
            registry_path,
            league=league,
            active_features=current_ordered,
            candidate_features=[],
        )
        return FeaturePolicyResult(
            approved_feature_columns=current_ordered,
            registry_path=str(registry_path),
            mode=mode_norm,
            added_features=sorted(current_set),
            removed_features=[],
            candidates_added=[],
            registry_updated=True,
            registry_created=True,
        )

    added = sorted(current_set - active_set)
    removed = sorted(active_set - current_set)

    if mode_norm == "production":
        if (added or removed) and not approve_changes:
            details = []
            if added:
                details.append(f"added={added}")
            if removed:
                details.append(f"removed={removed}")
            raise RuntimeError(
                "Feature contract drift detected in production mode ("
                + ", ".join(details)
                + "). Re-run with '--approve-feature-changes' to explicitly accept this contract update."
            )
        if added or removed:
            updated_candidates = sorted(candidate_set - current_set)
            _write_registry(
                registry_path,
                league=league,
                active_features=current_ordered,
                candidate_features=updated_candidates,
            )
            return FeaturePolicyResult(
                approved_feature_columns=current_ordered,
                registry_path=str(registry_path),
                mode=mode_norm,
                added_features=added,
                removed_features=removed,
                candidates_added=[],
                registry_updated=True,
                registry_created=False,
            )
        return FeaturePolicyResult(
            approved_feature_columns=current_ordered,
            registry_path=str(registry_path),
            mode=mode_norm,
            added_features=[],
            removed_features=[],
            candidates_added=[],
            registry_updated=False,
            registry_created=False,
        )

    # research mode: track newly discovered features as candidates, never block.
    newly_discovered = sorted((current_set - active_set) - candidate_set)
    if approve_changes:
        updated_candidates = sorted(candidate_set - current_set)
        _write_registry(
            registry_path,
            league=league,
            active_features=current_ordered,
            candidate_features=updated_candidates,
        )
        return FeaturePolicyResult(
            approved_feature_columns=current_ordered,
            registry_path=str(registry_path),
            mode=mode_norm,
            added_features=added,
            removed_features=removed,
            candidates_added=[],
            registry_updated=True,
            registry_created=False,
        )

    if newly_discovered:
        updated_candidates = sorted(candidate_set | set(newly_discovered))
        _write_registry(
            registry_path,
            league=league,
            active_features=existing_active,
            candidate_features=updated_candidates,
        )
        return FeaturePolicyResult(
            approved_feature_columns=current_ordered,
            registry_path=str(registry_path),
            mode=mode_norm,
            added_features=added,
            removed_features=removed,
            candidates_added=newly_discovered,
            registry_updated=True,
            registry_created=False,
        )

    return FeaturePolicyResult(
        approved_feature_columns=current_ordered,
        registry_path=str(registry_path),
        mode=mode_norm,
        added_features=added,
        removed_features=removed,
        candidates_added=[],
        registry_updated=False,
        registry_created=False,
    )

