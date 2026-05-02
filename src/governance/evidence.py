"""Governance metadata shared by validation, research, and promotion artifacts.

The MLB rebuild has two kinds of evidence that must never be confused:

* monograph-traceable model diagnostics and theory-compatible engineering
  checks, which support model-family validation; and
* betting overlays, which can be operationally useful but are not CAS theory
  proof.

This module centralizes those labels so validation contracts, tournament rows,
current-best payloads, and promotion summaries all speak the same small
vocabulary.  The functions are intentionally plain dictionaries because these
payloads are written to JSON, CSV, SQLite, and dashboard staging files.
"""

from __future__ import annotations

from typing import Any

from src.registry.models import get_model_registry_entry

GOVERNANCE_CONTRACT_VERSION = 1

DIRECT_THEORY_IMPLEMENTATION = "direct theory implementation"
THEORY_COMPATIBLE_ENGINEERING_SUPPORT = "theory-compatible engineering support"
BETTING_OVERLAY = "betting overlay"
DASHBOARD_REPORTING_LAYER = "dashboard/reporting layer"

THEORY_CORE_EVIDENCE_LANE = "theory_core_diagnostics"
THEORY_COMPATIBLE_ENGINEERING_LANE = "theory_compatible_engineering"
BETTING_OVERLAY_LANE = "betting_overlay"

MLB_PRODUCT_SCOPE = ("moneyline", "runline", "totals")
CAS_GOVERNING_SOURCES = (
    "statistical_theory/09_GLM_Generalized_Linear_Models_for_Insurance_Rating.pdf",
    "statistical_theory/10_Holmes_Casotto_Penalized_Regression_and_Lasso_Credibility_2025_Revision.pdf",
)

_BETTING_OVERLAY_MODELS = {"market_baseline", "intercept_only"}


def market_for_target(target_name: str | None) -> str:
    """Return the active betting market name for a target identifier."""

    token = str(target_name or "").strip().lower()
    if token.startswith("runline"):
        return "runline"
    if token.startswith("totals"):
        return "totals"
    return "moneyline"


def target_scope_fields(
    *,
    league: str = "MLB",
    target_name: str | None = None,
    target_col: str | None = None,
    market: str | None = None,
) -> dict[str, Any]:
    """Build the canonical MLB target-scope payload used by evidence artifacts."""

    resolved_target = str(target_name or "moneyline_home_win").strip()
    return {
        "league": str(league or "MLB").upper(),
        "target_name": resolved_target,
        "target_column": str(target_col or "home_win").strip(),
        "market": str(market or market_for_target(resolved_target)).strip(),
        "product_scope": list(MLB_PRODUCT_SCOPE),
    }


def _registry_model_fields(model_name: str) -> dict[str, Any] | None:
    key = str(model_name or "").strip()
    if not key:
        return None
    try:
        entry = get_model_registry_entry(key)
    except KeyError:
        return None
    if entry.lane == "core":
        theory_governance = DIRECT_THEORY_IMPLEMENTATION
        evidence_lane = THEORY_CORE_EVIDENCE_LANE
    else:
        theory_governance = THEORY_COMPATIBLE_ENGINEERING_SUPPORT
        evidence_lane = THEORY_COMPATIBLE_ENGINEERING_LANE
    return {
        "model_lane": entry.lane,
        "model_family": entry.family,
        "theory_classification": entry.theory_classification,
        "theory_governance": theory_governance,
        "evidence_lane": evidence_lane,
        "governing_sources": list(CAS_GOVERNING_SOURCES),
        "model_governance_note": entry.governance_note,
    }


def _fallback_model_fields(model_name: str) -> dict[str, Any]:
    token = str(model_name or "").strip()
    if token in _BETTING_OVERLAY_MODELS:
        return {
            "model_lane": "betting_overlay",
            "model_family": "market",
            "theory_classification": "experimental",
            "theory_governance": BETTING_OVERLAY,
            "evidence_lane": BETTING_OVERLAY_LANE,
            "governing_sources": [],
            "model_governance_note": "Betting overlay baseline; not a CAS theory-core model.",
        }
    return {
        "model_lane": "unknown",
        "model_family": "unknown",
        "theory_classification": "experimental",
        "theory_governance": THEORY_COMPATIBLE_ENGINEERING_SUPPORT,
        "evidence_lane": THEORY_COMPATIBLE_ENGINEERING_LANE,
        "governing_sources": [],
        "model_governance_note": "Unregistered comparison row; treat as non-default evidence until registered.",
    }


def model_evidence_fields(
    model_name: str,
    *,
    league: str = "MLB",
    target_name: str | None = None,
    target_col: str | None = None,
    market: str | None = None,
) -> dict[str, Any]:
    """Return deterministic governance fields for one model evidence row."""

    fields = _registry_model_fields(model_name) or _fallback_model_fields(model_name)
    fields["governance_contract_version"] = GOVERNANCE_CONTRACT_VERSION
    fields["target_scope"] = target_scope_fields(
        league=league,
        target_name=target_name,
        target_col=target_col,
        market=market,
    )
    return fields
