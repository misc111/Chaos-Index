"""Validation-task governance labels for the MLB evidence contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.governance.evidence import (
    BETTING_OVERLAY,
    BETTING_OVERLAY_LANE,
    DASHBOARD_REPORTING_LAYER,
    DIRECT_THEORY_IMPLEMENTATION,
    GOVERNANCE_CONTRACT_VERSION,
    THEORY_COMPATIBLE_ENGINEERING_LANE,
    THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
    THEORY_CORE_EVIDENCE_LANE,
    model_evidence_fields,
    target_scope_fields,
)


@dataclass(frozen=True, slots=True)
class ValidationTaskGovernance:
    """Canonical evidence lane labels for one validation task."""

    validation_lane: str
    theory_governance: str


ENGINEERING_TASK_GOVERNANCE = ValidationTaskGovernance(
    validation_lane=THEORY_COMPATIBLE_ENGINEERING_LANE,
    theory_governance=THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
)
THEORY_CORE_TASK_GOVERNANCE = ValidationTaskGovernance(
    validation_lane=THEORY_CORE_EVIDENCE_LANE,
    theory_governance=DIRECT_THEORY_IMPLEMENTATION,
)
BETTING_OVERLAY_TASK_GOVERNANCE = ValidationTaskGovernance(
    validation_lane=BETTING_OVERLAY_LANE,
    theory_governance=BETTING_OVERLAY,
)


def split_label_for_context(ctx: Any) -> str:
    """Return the resolved split label used by validation artifacts and SQLite."""

    split_plan = getattr(ctx, "split_plan", None)
    resolved_mode = str(getattr(split_plan, "resolved_mode", "") or "").strip()
    resolved_method = str(getattr(split_plan, "resolved_method", "") or "").strip()
    if resolved_mode and resolved_method:
        return f"{resolved_mode}:{resolved_method}"
    return resolved_mode or resolved_method or "unknown_split"


def target_scope_for_context(ctx: Any) -> dict[str, Any]:
    """Resolve the MLB target scope from run payload and validation context."""

    run_payload = getattr(ctx, "run_payload", {}) or {}
    return target_scope_fields(
        league=str(getattr(ctx, "league", "MLB") or "MLB"),
        target_name=run_payload.get("target_name"),
        target_col=run_payload.get("target_col"),
        market=run_payload.get("market"),
    )


def validation_record_governance_fields(ctx: Any, *, task_name: str, validation_lane: str, theory_governance: str) -> dict[str, Any]:
    """Build task-record fields that separate theory, engineering, and overlays."""

    metadata = getattr(ctx, "glm_validation_metadata", None)
    model_name = str(
        getattr(metadata, "model_key", "")
        or getattr(metadata, "model_name", "")
        or (getattr(ctx, "run_payload", {}) or {}).get("glm_primary_model")
        or ""
    )
    target_scope = target_scope_for_context(ctx)
    model_fields = model_evidence_fields(
        model_name,
        league=str(target_scope["league"]),
        target_name=str(target_scope["target_name"]),
        target_col=str(target_scope["target_column"]),
        market=str(target_scope["market"]),
    )
    return {
        "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
        "validation_lane": str(validation_lane),
        "theory_governance": str(theory_governance),
        "dashboard_output_lane": DASHBOARD_REPORTING_LAYER,
        "split_label": split_label_for_context(ctx),
        "target_scope": target_scope,
        "model_lane": str(model_fields.get("model_lane") or ""),
        "theory_classification": str(model_fields.get("theory_classification") or ""),
        "governing_sources": list(model_fields.get("governing_sources") or []),
        "validation_task_name": str(task_name),
    }
