import assert from "node:assert/strict";
import test from "node:test";

import { buildHomeModelEvidenceBadge, indexCurrentBestTopModels } from "./home-model-evidence";
import type { HomeEvidencePayload } from "./home-model-evidence";

test("landing cards explain live testing without insider research labels", () => {
  const evidence: HomeEvidencePayload = {
    latest_artifact_role: "latest_research_recommendation",
    promotion_eligible: false,
    production_ready: false,
    evidence_status: {
      evidence_stage: "research_only",
      blocked_reasons: ["candidate_comparison_is_research_only"],
    },
    current_best_top_models: [{ model_name: "dglm_margin", rank: 1 }],
  };

  const badge = buildHomeModelEvidenceBadge("dglm_margin", evidence, indexCurrentBestTopModels(evidence.current_best_top_models));

  assert.equal(badge.tone, "research");
  assert.equal(badge.reason, "Showing promise, but not trusted for picks yet.");
});

test("landing cards explain review-ready status in plain language", () => {
  const evidence: HomeEvidencePayload = {
    latest_artifact_role: "latest_promotion_eligible_evidence",
    promotion_eligible: true,
    production_ready: false,
    current_best_top_models: [{ model_name: "glm_ridge", rank: 2 }],
  };

  const badge = buildHomeModelEvidenceBadge("glm_ridge", evidence, indexCurrentBestTopModels(evidence.current_best_top_models));

  assert.equal(badge.tone, "review");
  assert.equal(badge.reason, "In the review mix.");
});

test("landing cards show blocked status with the first short evidence reason", () => {
  const evidence: HomeEvidencePayload = {
    latest_artifact_role: "blocked_promotion_evidence",
    promotion_eligible: false,
    production_ready: false,
    evidence_status: {
      evidence_stage: "promotion_eligible",
      readiness_blocked_reasons: ["not_full_immutable_pregame_mlb_ledger"],
    },
    current_best_top_models: [{ model_name: "glm_lasso", rank: 3 }],
  };

  const badge = buildHomeModelEvidenceBadge("glm_lasso", evidence, indexCurrentBestTopModels(evidence.current_best_top_models));

  assert.equal(badge.tone, "blocked");
  assert.equal(badge.reason, "Needs a clean pregame track record.");
});

test("landing cards keep non-ranked trainable models research-only", () => {
  const evidence: HomeEvidencePayload = {
    latest_artifact_role: "latest_research_recommendation",
    current_best_top_models: [{ model_name: "dglm_margin", rank: 1 }],
  };

  const badge = buildHomeModelEvidenceBadge("glm_elastic_net", evidence, indexCurrentBestTopModels(evidence.current_best_top_models));

  assert.equal(badge.tone, "research");
  assert.equal(badge.reason, "Waiting for enough fresh game proof.");
});
