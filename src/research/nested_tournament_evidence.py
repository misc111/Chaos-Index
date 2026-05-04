"""Evidence-status helpers for nested MLB tournament artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def nested_tournament_evidence_fields(
    *,
    run_id: str,
    source_kind: str,
    artifact_root: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Return research-only latest-pointer fields for nested tournament payloads."""

    fixture_demo_smoke = any(token in f"{run_id} {artifact_root}".lower() for token in ("fixture", "demo", "smoke"))
    evidence_stage = "fixture_demo_smoke" if fixture_demo_smoke else "research_only"
    return {
        "latest_artifact_role": "latest_research_recommendation",
        "evidence_stage": evidence_stage,
        "evidence_status": {
            "evidence_stage": evidence_stage,
            "label": "nested tournament research evidence",
            "latest_artifact_role": "latest_research_recommendation",
            "pointer_semantics": (
                "This pointer identifies the latest nested tournament research evidence. "
                "It is not full-ledger promotion-eligible evidence."
            ),
            "promotion_eligible": False,
            "production_grade": False,
            "production_ready": False,
            "fixture_demo_smoke": fixture_demo_smoke,
            "full_immutable_pregame_ledger": False,
            "blocked_reasons": ["nested_tournament_research_only", "not_full_immutable_pregame_mlb_ledger"],
        },
        "promotion_eligible": False,
        "production_ready": False,
        "latest_research_recommendation": {
            "run_id": run_id,
            "source_kind": source_kind,
            "pointer_semantics": "Latest nested tournament research evidence only; not promotion-eligible evidence.",
            "artifact_path": str(summary_path),
        },
        "latest_promotion_eligible_evidence": None,
    }
