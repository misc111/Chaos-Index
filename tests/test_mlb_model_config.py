from __future__ import annotations

import pytest
import yaml

from src.registry.models import model_manifest_payload
from src.training.model_catalog import (
    CAS_CORE_FAMILY_CATALOG,
    PENALIZED_CORE_MODEL_NAMES,
    PLANNED_CREDIBILITY_MODEL_NAMES,
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
        "two_stage",
    }


def test_mlb_lasso_credibility_models_are_trainable_but_not_default_enabled() -> None:
    assert normalize_selected_models(["market_offset_lasso"]) == ["glm_lasso_market_credibility"]
    assert normalize_selected_models(["prior_offset_lasso"]) == ["glm_lasso_prior_credibility"]


def test_mlb_non_cas_challengers_are_quarantined_to_experimental_lane() -> None:
    payload = yaml.safe_load(open("configs/model_feature_map_mlb.yaml").read())

    experimental = set((payload or {}).get("experimental_models", {}))
    assert experimental == {"gbdt", "rf", "bayes_bt_state_space", "nn_mlp"}
    assert not experimental.intersection(PLANNED_CREDIBILITY_MODEL_NAMES)


def test_mlb_model_manifest_spells_out_core_and_challenger_lanes() -> None:
    manifest = model_manifest_payload()

    assert manifest["primary_lane"] == "core"
    assert manifest["default_training_models"][0:3] == ["glm_ridge", "glm_elastic_net", "glm_lasso"]
    assert "glm_lasso_market_credibility" in manifest["core_models"]
    assert "glm_lasso_prior_credibility" in manifest["core_models"]
    assert "glm_lasso_market_credibility" not in manifest["default_training_models"]
    assert "glm_lasso_prior_credibility" not in manifest["default_training_models"]
    assert manifest["penalized_core_models"] == PENALIZED_CORE_MODEL_NAMES
    assert manifest["baseline_models"] == [
        "elo_baseline",
        "dynamic_rating",
        "simulation_first",
    ]
    assert set(manifest["experimental_models"]) == {
        "gbdt",
        "rf",
        "bayes_bt_state_space",
        "bayes_goals",
        "nn_mlp",
    }
    assert manifest["lane_labels"]["core"] == "CAS core lane"
    assert manifest["planned_credibility_model_keys"] == PLANNED_CREDIBILITY_MODEL_NAMES
    assert manifest["planned_credibility_models"]["glm_lasso_market_credibility"]["complement_column"] == "market_offset_logit"
    assert manifest["planned_model_aliases"]["market_offset_lasso"] == "glm_lasso_market_credibility"
    assert manifest["cas_core_families"] == CAS_CORE_FAMILY_CATALOG
    assert manifest["models"]["glm_lasso_market_credibility"]["default_enabled"] is False
    assert manifest["models"]["glm_lasso_prior_credibility"]["default_enabled"] is False
    assert manifest["models"]["gbdt"]["default_enabled"] is False


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
