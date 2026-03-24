from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


GLM_CANDIDATE_MODELS = ("glm_ridge", "glm_elastic_net", "glm_lasso", "glm_vanilla")


@dataclass(frozen=True, slots=True)
class StructuredGLMSelection:
    spec_path: Path
    experiment_name: str
    slate_name: str
    width_variant: str | None
    requested_feature_count: int
    available_feature_count: int
    missing_feature_count: int
    features: tuple[str, ...]

    def feature_overrides(self) -> dict[str, list[str]]:
        return {model_name: list(self.features) for model_name in GLM_CANDIDATE_MODELS}

    def summary_line(self) -> str:
        variant_note = f", width variant `{self.width_variant}`" if self.width_variant else ""
        missing_note = f", missing_after_pool={self.missing_feature_count}" if self.missing_feature_count else ""
        return (
            f"structured GLM experiment `{self.experiment_name}` from `{self.spec_path}` "
            f"(slate `{self.slate_name}`{variant_note}; requested={self.requested_feature_count}, "
            f"usable={self.available_feature_count}{missing_note})"
        )


def _resolve_spec_path(spec_path: str | Path) -> Path:
    path = Path(spec_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _parse_feature_count(value: Any, *, field_name: str) -> int:
    if isinstance(value, int):
        count = int(value)
    elif isinstance(value, dict):
        count = int(value.get("feature_count", 0))
    else:
        count = 0
    if count <= 0:
        raise ValueError(f"Structured GLM spec field `{field_name}` must define a positive feature count")
    return count


def load_structured_glm_selection(
    *,
    league: str,
    available_features: list[str],
    spec_path: str | Path | None,
    slate_name: str | None = None,
    width_variant: str | None = None,
) -> StructuredGLMSelection | None:
    if not spec_path:
        return None

    league_code = str(league or "").strip().upper()
    if league_code != "NBA":
        raise ValueError("Structured GLM experiment specs are research-only and currently supported for NBA only.")

    path = _resolve_spec_path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Structured GLM experiment spec not found: {path}")

    payload = yaml.safe_load(path.read_text()) or {}
    spec_league = str(payload.get("league", league_code)).strip().upper()
    if spec_league and spec_league != league_code:
        raise ValueError(
            f"Structured GLM spec league mismatch: file declares `{spec_league}` but runtime league is `{league_code}`."
        )

    slates = payload.get("slates")
    if not isinstance(slates, dict) or not slates:
        raise ValueError(f"Structured GLM spec `{path}` must include a non-empty `slates` mapping.")

    selected_slate = str(slate_name or payload.get("default_slate") or "").strip()
    if not selected_slate:
        selected_slate = str(next(iter(slates)))
    if selected_slate not in slates:
        raise ValueError(
            f"Structured GLM spec `{path}` does not contain slate `{selected_slate}`. "
            f"Valid={sorted(str(name) for name in slates.keys())}"
        )

    slate_payload = slates[selected_slate]
    if not isinstance(slate_payload, dict):
        raise ValueError(f"Structured GLM slate `{selected_slate}` must map to a dictionary payload.")
    feature_order = [str(feature).strip() for feature in slate_payload.get("feature_order", []) if str(feature).strip()]
    if not feature_order:
        raise ValueError(f"Structured GLM slate `{selected_slate}` must define a non-empty `feature_order` list.")

    variant_payload = slate_payload.get("width_variants", {})
    if variant_payload is None:
        variant_payload = {}
    if not isinstance(variant_payload, dict):
        raise ValueError(f"Structured GLM slate `{selected_slate}` has an invalid `width_variants` payload.")

    selected_variant = str(
        width_variant
        or slate_payload.get("default_width_variant")
        or payload.get("default_width_variant")
        or ""
    ).strip()
    requested_feature_count = len(feature_order)
    if selected_variant:
        if selected_variant not in variant_payload:
            raise ValueError(
                f"Structured GLM slate `{selected_slate}` does not contain width variant `{selected_variant}`. "
                f"Valid={sorted(str(name) for name in variant_payload.keys())}"
            )
        requested_feature_count = _parse_feature_count(
            variant_payload[selected_variant],
            field_name=f"slates.{selected_slate}.width_variants.{selected_variant}",
        )
    requested_feature_count = min(requested_feature_count, len(feature_order))
    requested_features = feature_order[:requested_feature_count]

    available_set = {str(column) for column in available_features}
    resolved_features = [feature for feature in requested_features if feature in available_set]
    missing_feature_count = len(requested_features) - len(resolved_features)
    if not resolved_features:
        raise ValueError(
            f"Structured GLM selection `{selected_slate}` from `{path}` resolved zero usable features "
            f"after applying the current feature pool."
        )

    experiment_name = str(payload.get("experiment_name", path.stem)).strip() or path.stem
    return StructuredGLMSelection(
        spec_path=path,
        experiment_name=experiment_name,
        slate_name=selected_slate,
        width_variant=selected_variant or None,
        requested_feature_count=len(requested_features),
        available_feature_count=len(resolved_features),
        missing_feature_count=missing_feature_count,
        features=tuple(resolved_features),
    )
