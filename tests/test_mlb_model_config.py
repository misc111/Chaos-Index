from __future__ import annotations

import yaml

from src.registry.models import model_manifest_payload
from src.training.model_catalog import (
    BASELINE_MODEL_NAMES,
    CAS_CORE_FAMILY_CATALOG,
    CORE_MODEL_NAMES,
    DEFAULT_MODEL_NAMES,
    EXPERIMENTAL_MODEL_NAMES,
    GOVERNANCE_COMPARISON_GROUPS,
    MODEL_SELECTION_GROUP_ALIASES,
    PENALIZED_CORE_MODEL_NAMES,
    PLANNED_CREDIBILITY_MODEL_NAMES,
    THEORY_EXTENSION_MODEL_NAMES,
    normalize_selected_models,
)
from src.training.model_feature_guardrails import load_model_feature_guardrails
from src.training.model_feature_research import load_model_feature_map


def test_mlb_active_model_feature_map_stays_cas_core_by_default() -> None:
    active_map = load_model_feature_map("MLB")

    assert set(active_map) == {
        "glm_ridge",
        "glm_elastic_net",
        "glm_lasso",
        "glm_lasso_market_credibility",
        "glm_lasso_prior_credibility",
    }


def test_mlb_lasso_credibility_models_are_trainable_but_not_default_enabled() -> None:
    assert normalize_selected_models(["market_offset_lasso"]) == ["glm_lasso_market_credibility"]
    assert normalize_selected_models(["prior_offset_lasso"]) == ["glm_lasso_prior_credibility"]


def test_mlb_group_tokens_expand_to_governance_aligned_model_sets() -> None:
    assert normalize_selected_models(["core_default"]) == DEFAULT_MODEL_NAMES
    assert normalize_selected_models(["core"]) == CORE_MODEL_NAMES
    assert normalize_selected_models(["challengers"]) == THEORY_EXTENSION_MODEL_NAMES
    assert normalize_selected_models(["theory_core_opt_in"]) == PLANNED_CREDIBILITY_MODEL_NAMES
    assert normalize_selected_models(["theory_compatible_extensions"]) == THEORY_EXTENSION_MODEL_NAMES
    assert normalize_selected_models(["core_default", "theory_core_opt_in"]) == [
        *DEFAULT_MODEL_NAMES,
        *PLANNED_CREDIBILITY_MODEL_NAMES,
    ]
    assert MODEL_SELECTION_GROUP_ALIASES["experimental_challengers"] == tuple(EXPERIMENTAL_MODEL_NAMES)


def test_retired_group_tokens_fail_closed_under_monograph_only_lane() -> None:
    try:
        normalize_selected_models(["baseline"])
    except ValueError:
        pass
    else:
        raise AssertionError("baseline should be retired from the active monograph-only lane")

    try:
        normalize_selected_models(["experimental"])
    except ValueError:
        pass
    else:
        raise AssertionError("experimental should be retired from the active monograph-only lane")


def test_mlb_non_cas_challengers_are_quarantined_to_experimental_lane() -> None:
    payload = yaml.safe_load(open("configs/model_feature_map_mlb.yaml").read())

    experimental = set((payload or {}).get("experimental_models", {}))
    assert experimental == set()
    assert not experimental.intersection(PLANNED_CREDIBILITY_MODEL_NAMES)


def test_mlb_model_manifest_spells_out_core_and_challenger_lanes() -> None:
    manifest = model_manifest_payload()

    assert manifest["primary_lane"] == "core"
    assert manifest["default_training_models"] == ["glm_ridge", "glm_elastic_net", "glm_lasso", "glm_vanilla"]
    assert "glm_lasso_market_credibility" in manifest["core_models"]
    assert "glm_lasso_prior_credibility" in manifest["core_models"]
    assert "glm_lasso_market_credibility" not in manifest["default_training_models"]
    assert "glm_lasso_prior_credibility" not in manifest["default_training_models"]
    assert manifest["theory_extension_models"] == THEORY_EXTENSION_MODEL_NAMES
    assert manifest["penalized_core_models"] == PENALIZED_CORE_MODEL_NAMES
    assert manifest["baseline_models"] == []
    assert manifest["experimental_models"] == []
    assert manifest["lane_labels"]["core"] == "CAS core lane"
    assert manifest["lane_labels"]["extension"] == "Theory-compatible extension lane"
    assert manifest["planned_credibility_model_keys"] == PLANNED_CREDIBILITY_MODEL_NAMES
    assert manifest["planned_credibility_models"]["glm_lasso_market_credibility"]["complement_column"] == "market_offset_logit"
    assert manifest["planned_model_aliases"]["market_offset_lasso"] == "glm_lasso_market_credibility"
    assert manifest["cas_core_families"] == CAS_CORE_FAMILY_CATALOG
    assert manifest["models"]["glm_lasso_market_credibility"]["default_enabled"] is False
    assert manifest["models"]["glm_lasso_prior_credibility"]["default_enabled"] is False
    assert manifest["models"]["gam_spline"]["default_enabled"] is False
    assert manifest["models"]["glmm_logit"]["default_enabled"] is False


def test_mlb_credibility_catalog_and_guardrails_stay_coherent() -> None:
    payload = yaml.safe_load(open("configs/model_feature_map_mlb.yaml").read())
    planned = (payload or {}).get("planned_credibility_models", {})
    theory_families = (payload or {}).get("core_theory_families", {})
    guardrails = load_model_feature_guardrails("MLB")

    assert theory_families["penalized_glm"]["model_keys"] == PENALIZED_CORE_MODEL_NAMES
    assert theory_families["lasso_credibility"]["model_keys"] == PLANNED_CREDIBILITY_MODEL_NAMES
    assert set(planned) == set(PLANNED_CREDIBILITY_MODEL_NAMES)
    assert planned["glm_lasso_market_credibility"]["driver_features"] == payload["models"]["glm_lasso"]["active_features"]
    assert planned["glm_lasso_prior_credibility"]["driver_features"] == payload["models"]["glm_lasso"]["active_features"]
    assert payload["models"]["glm_lasso_market_credibility"]["complement_column"] == "market_offset_logit"
    assert payload["models"]["glm_lasso_prior_credibility"]["complement_column"] == "prior_model_offset_logit"
    assert "market_offset_logit" in guardrails["glm_lasso_market_credibility"]["blocked_features"]
    assert "prior_model_offset_logit" in guardrails["glm_lasso_prior_credibility"]["blocked_features"]


def test_mlb_comparison_profiles_align_with_catalog_governance() -> None:
    feature_map_payload = yaml.safe_load(open("configs/model_feature_map_mlb.yaml").read()) or {}
    guardrails_payload = yaml.safe_load(open("configs/model_feature_guardrails_mlb.yaml").read()) or {}
    comparison_profiles = feature_map_payload.get("comparison_reporting_profiles", {})
    policy_profiles = guardrails_payload.get("comparison_reporting_guardrails", {})

    assert set(comparison_profiles) == set(GOVERNANCE_COMPARISON_GROUPS)
    assert set(policy_profiles) == set(GOVERNANCE_COMPARISON_GROUPS)
    assert comparison_profiles["theory_core_default"]["model_keys"] == DEFAULT_MODEL_NAMES
    assert comparison_profiles["theory_core_opt_in"]["model_keys"] == PLANNED_CREDIBILITY_MODEL_NAMES
    assert comparison_profiles["theory_compatible_extensions"]["model_keys"] == THEORY_EXTENSION_MODEL_NAMES
    assert comparison_profiles["baseline_references"]["model_keys"] == BASELINE_MODEL_NAMES
    assert comparison_profiles["experimental_challengers"]["model_keys"] == []
    assert comparison_profiles["experimental_challengers"]["champion_eligible"] is False
    assert policy_profiles["theory_core_default"]["champion_eligible"] is True
    assert policy_profiles["theory_compatible_extensions"]["promotion_gate"] == "require_extension_evidence_packet"
    assert policy_profiles["experimental_challengers"]["promotion_gate"] == "retired"
