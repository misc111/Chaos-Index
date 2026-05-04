import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { loadModelFeatureMapFromArtifacts } from "./model-feature-map";

async function writeJson(filePath: string, payload: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

test("loads fitted model feature sets from the latest MLB run artifact", async () => {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), "sportsmodeling-feature-map-"));
  const olderRun = path.join(repoRoot, "artifacts", "models", "older", "run_payload.json");
  const latestRun = path.join(repoRoot, "artifacts", "models", "latest", "run_payload.json");

  await writeJson(olderRun, {
    league: "MLB",
    model_run_id: "run_older",
    model_feature_columns: {
      glm_ridge: ["stale_feature"],
    },
    model_artifacts: [
      {
        model_name: "glm_ridge",
        feature_columns: ["stale_feature"],
        fit_summary: { active_features: ["stale_feature"], active_parameter_count: 1 },
      },
    ],
  });
  await writeJson(latestRun, {
    league: "MLB",
    feature_set_version: "fset_latest",
    model_run_id: "run_latest",
    model_feature_columns: {
      glm_ridge: ["rest_diff", "travel_diff"],
      glm_lasso: ["diff_starting_pitcher_quality", "home_field_advantage"],
    },
    model_artifacts: [
      {
        model_name: "glm_ridge",
        distribution: "binomial",
        link_function: "logit",
        feature_columns: ["rest_diff", "travel_diff"],
        fit_summary: {
          active_features: ["rest_diff"],
          active_parameter_count: 1,
          penalty_family: "ridge",
          active_coefficient_summary: [{ feature: "rest_diff", active_rank: 1, direction: "positive" }],
        },
      },
      {
        model_name: "glm_lasso",
        feature_columns: ["diff_starting_pitcher_quality", "home_field_advantage"],
        fit_summary: {
          active_features: [],
          active_parameter_count: 0,
          coefficient_path_metadata: {
            top_feature_names: ["diff_starting_pitcher_quality", "home_field_advantage"],
          },
        },
      },
    ],
  });

  const stale = new Date("2026-05-01T00:00:00Z");
  const fresh = new Date("2026-05-04T00:00:00Z");
  await fs.utimes(path.dirname(olderRun), stale, stale);
  await fs.utimes(path.dirname(latestRun), fresh, fresh);
  await fs.utimes(olderRun, stale, stale);
  await fs.utimes(latestRun, fresh, fresh);

  const payload = await loadModelFeatureMapFromArtifacts("MLB", { repoRoot });

  assert.equal(payload.source, "latest_fit_artifacts");
  assert.equal(payload.model_run_id, "run_latest");
  assert.equal(payload.feature_set_version, "fset_latest");
  assert.deepEqual(payload.models.glm_ridge.active_features, ["rest_diff", "travel_diff"]);
  assert.equal(payload.models.glm_ridge.active_feature_count, 2);
  assert.deepEqual(payload.models.glm_ridge.fit_active_features, ["rest_diff"]);
  assert.equal(payload.models.glm_ridge.fit_active_feature_count, 1);
  assert.equal(payload.models.glm_ridge.penalty_family, "ridge");
  assert.equal(payload.models.glm_ridge.theory_trace?.governance, "direct theory implementation");
  assert.deepEqual(payload.models.glm_lasso.active_features, ["diff_starting_pitcher_quality", "home_field_advantage"]);
  assert.deepEqual(payload.models.glm_lasso.coefficient_top_features, [
    "diff_starting_pitcher_quality",
    "home_field_advantage",
  ]);
});

test("loads lasso credibility driver features from standalone credibility sidecars", async () => {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), "sportsmodeling-cred-feature-map-"));
  const metadataPath = path.join(
    repoRoot,
    "artifacts",
    "models",
    "credibility",
    "glm_lasso_market_credibility_credibility_metadata.json"
  );

  await writeJson(metadataPath, {
    complement_kind: "market",
    complement_column: "market_offset_logit",
    complement_label: "Vig-free market logit",
    offset_scale: "logit",
    driver_features: ["diff_starting_pitcher_quality", "rest_diff"],
    relativity_summary: {
      active_driver_feature_count: 1,
      driver_feature_count: 2,
      active_coefficient_summary: [{ feature: "rest_diff", active_rank: 1, direction: "positive" }],
      coefficient_path_metadata: {
        top_feature_names: ["rest_diff", "diff_starting_pitcher_quality"],
      },
    },
  });

  const payload = await loadModelFeatureMapFromArtifacts("MLB", { repoRoot });

  assert.deepEqual(payload.models.glm_lasso_market_credibility.active_features, [
    "diff_starting_pitcher_quality",
    "rest_diff",
  ]);
  assert.equal(payload.models.glm_lasso_market_credibility.complement_kind, "market");
  assert.equal(payload.models.glm_lasso_market_credibility.complement_column, "market_offset_logit");
  assert.deepEqual(payload.models.glm_lasso_market_credibility.fit_active_features, ["rest_diff"]);
  assert.equal(payload.models.glm_lasso_market_credibility.fit_active_feature_count, 1);
  assert.deepEqual(payload.models.glm_lasso_market_credibility.coefficient_top_features, [
    "rest_diff",
    "diff_starting_pitcher_quality",
  ]);
});

test("resolves artifacts when the process cwd is already the repo root", async () => {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), "sportsmodeling-cwd-feature-map-"));
  const originalCwd = process.cwd();
  const runPayload = path.join(repoRoot, "artifacts", "models", "latest", "run_payload.json");

  await writeJson(runPayload, {
    league: "MLB",
    model_run_id: "run_from_repo_root",
    model_artifacts: [
      {
        model_name: "glm_ridge",
        feature_columns: ["rest_diff"],
        fit_summary: { active_features: ["rest_diff"] },
      },
    ],
  });

  try {
    process.chdir(repoRoot);
    const payload = await loadModelFeatureMapFromArtifacts("MLB");
    assert.equal(payload.model_run_id, "run_from_repo_root");
    assert.deepEqual(payload.models.glm_ridge.active_features, ["rest_diff"]);
  } finally {
    process.chdir(originalCwd);
  }
});

test("tolerates NaN sentinels in Python-written run payloads", async () => {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), "sportsmodeling-nan-feature-map-"));
  const runPayload = path.join(repoRoot, "artifacts", "models", "latest", "run_payload.json");
  await fs.mkdir(path.dirname(runPayload), { recursive: true });
  await fs.writeFile(
    runPayload,
    `{
      "league": "MLB",
      "model_run_id": "run_with_nan",
      "irrelevant_metric": NaN,
      "model_artifacts": [
        {
          "model_name": "glm_ridge",
          "feature_columns": ["rest_diff"],
          "fit_summary": {
            "active_features": ["rest_diff"],
            "coefficient_path_metadata": { "delta_coef_l2": NaN }
          }
        }
      ]
    }\n`,
    "utf8"
  );

  const payload = await loadModelFeatureMapFromArtifacts("MLB", { repoRoot });

  assert.equal(payload.model_run_id, "run_with_nan");
  assert.deepEqual(payload.models.glm_ridge.active_features, ["rest_diff"]);
});
