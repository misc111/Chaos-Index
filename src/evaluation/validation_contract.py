from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.registry.models import get_model_registry_entry
from src.training.contract_builders import build_lasso_credibility_metadata, build_penalty_selection
from src.training.model_catalog import MODEL_ALIASES
from src.training.contracts import LassoCredibilityMetadata, PenaltySelection, ValidationArtifactRecord


@dataclass(frozen=True, slots=True)
class ValidationModelMetadata:
    model_name: str
    model_key: str = ""
    model_family: str = ""
    model_lane: str = ""
    model_governance_note: str = ""
    penalty: PenaltySelection | None = None
    credibility: LassoCredibilityMetadata | None = None

    @property
    def uses_penalized_glm(self) -> bool:
        return self.penalty is not None

    def to_summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.model_name:
            payload["model_name"] = self.model_name
        if self.model_key:
            payload["model_key"] = self.model_key
        if self.model_family:
            payload["model_family"] = self.model_family
        if self.model_lane:
            payload["model_lane"] = self.model_lane
        if self.model_governance_note:
            payload["model_governance_note"] = self.model_governance_note
        if self.penalty is not None:
            payload["penalty"] = self.penalty.to_dict()
        if self.credibility is not None:
            payload["credibility"] = self.credibility.to_dict()
        return payload


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _resolve_registry_payload(model_name: str) -> dict[str, str]:
    token = str(model_name or "").strip().lower()
    if not token:
        return {}
    model_key = MODEL_ALIASES.get(token, token)
    try:
        entry = get_model_registry_entry(model_key)
    except KeyError:
        return {"model_key": model_key} if model_key else {}
    return {
        "model_key": model_key,
        "model_family": str(entry.family),
        "model_lane": str(entry.lane),
        "model_governance_note": str(entry.governance_note),
    }


def _extract_credibility_payload(
    run_payload: Mapping[str, Any],
    *,
    model_name: str,
) -> dict[str, Any] | None:
    for key in ("credibility_metadata_by_model", "model_credibility_by_model"):
        mapping = run_payload.get(key)
        if isinstance(mapping, Mapping):
            candidate = _mapping_or_none(mapping.get(model_name))
            if candidate:
                return candidate

    for key in ("credibility_metadata", "validation_credibility_metadata"):
        candidate = _mapping_or_none(run_payload.get(key))
        if candidate:
            return candidate

    model_artifacts: list[Any] = []
    for key in ("model_artifacts",):
        value = run_payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            model_artifacts.extend(list(value))
    model_run_contract = run_payload.get("model_run_contract")
    if isinstance(model_run_contract, Mapping):
        value = model_run_contract.get("model_artifacts")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            model_artifacts.extend(list(value))

    for artifact in model_artifacts:
        if not isinstance(artifact, Mapping):
            continue
        if str(artifact.get("model_name") or "").strip() != model_name:
            continue
        candidate = _mapping_or_none(artifact.get("credibility"))
        if candidate:
            return candidate

    return None


def resolve_validation_model_metadata(
    *,
    run_payload: Mapping[str, Any],
    model: object | None,
    model_name: str | None = None,
) -> ValidationModelMetadata:
    resolved_model_name = str(
        model_name or getattr(model, "model_name", "") or run_payload.get("glm_primary_model") or ""
    ).strip()
    glm_tuning_by_model = run_payload.get("glm_tuning_by_model")
    penalty_tuning: Mapping[str, Any] | None = None
    if isinstance(glm_tuning_by_model, Mapping) and resolved_model_name:
        candidate = glm_tuning_by_model.get(resolved_model_name)
        if isinstance(candidate, Mapping):
            penalty_tuning = candidate
    if penalty_tuning is None:
        candidate = run_payload.get("glm_tuning")
        if isinstance(candidate, Mapping):
            penalty_tuning = candidate

    registry_payload = _resolve_registry_payload(resolved_model_name)
    credibility_payload = (
        _extract_credibility_payload(run_payload, model_name=resolved_model_name) if resolved_model_name else None
    )
    return ValidationModelMetadata(
        model_name=resolved_model_name,
        model_key=registry_payload.get("model_key", ""),
        model_family=registry_payload.get("model_family", ""),
        model_lane=registry_payload.get("model_lane", ""),
        model_governance_note=registry_payload.get("model_governance_note", ""),
        penalty=build_penalty_selection(resolved_model_name, tuning=penalty_tuning, model=model)
        if resolved_model_name
        else None,
        credibility=build_lasso_credibility_metadata(
            resolved_model_name,
            model=model,
            credibility_payload=credibility_payload,
        )
        if resolved_model_name
        else None,
    )


def build_validation_artifact_record(
    *,
    task_name: str,
    family: str,
    applicability: str,
    artifacts: Sequence[str],
    summary: Mapping[str, Any] | None = None,
) -> ValidationArtifactRecord:
    return ValidationArtifactRecord(
        task_name=str(task_name),
        family=str(family),
        applicability=str(applicability),
        artifacts=[str(path) for path in artifacts if str(path).strip()],
        summary=dict(summary or {}),
    )
