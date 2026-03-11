import assert from "node:assert/strict";
import test from "node:test";

import type { EnsembleSnapshotRunMetadata } from "@/lib/ensemble-snapshot-replay";
import { buildSnapshotActivationCutoffs } from "@/lib/server/services/performance";

function buildMetadata(
  overrides: Partial<EnsembleSnapshotRunMetadata>
): EnsembleSnapshotRunMetadata {
  return {
    model_name: "ensemble",
    model_run_id: "run_default",
    ensemble_model_run_id: "run_default__ensemble",
    finalized_at_utc: "2026-03-05T22:20:41Z",
    finalized_date_central: "2026-03-05",
    snapshot_id: "snapshot_default",
    artifact_path: "artifacts/models/default",
    feature_set_version: "fset_default",
    calibration_fingerprint: "fingerprint_default",
    feature_columns: ["rest_diff"],
    feature_metadata: null,
    params: null,
    metrics: null,
    tuning: null,
    selected_models: ["rf"],
    ensemble_component_columns: ["rf"],
    demoted_models: [],
    stack_base_columns: ["rf"],
    glm_feature_columns: ["rest_diff"],
    model_feature_columns: { rf: ["rest_diff"] },
    component_models: [],
    model_commit: null,
    commit_window: [],
    ...overrides,
  };
}

test("buildSnapshotActivationCutoffs preserves pregame snapshots and adds end-of-day cutoffs for late or off-day recalibrations", () => {
  const metadataById = new Map<string, EnsembleSnapshotRunMetadata>([
    [
      "run_pregame",
      buildMetadata({
        model_run_id: "run_pregame",
        ensemble_model_run_id: "run_pregame__ensemble",
        finalized_at_utc: "2026-03-05T22:20:41Z",
        finalized_date_central: "2026-03-05",
      }),
    ],
    [
      "run_eod",
      buildMetadata({
        model_run_id: "run_eod",
        ensemble_model_run_id: "run_eod__ensemble",
        finalized_at_utc: "2026-03-06T04:30:00Z",
        finalized_date_central: "2026-03-05",
        calibration_fingerprint: "fingerprint_eod",
      }),
    ],
    [
      "run_offday",
      buildMetadata({
        model_run_id: "run_offday",
        ensemble_model_run_id: "run_offday__ensemble",
        finalized_at_utc: "2026-03-10T03:00:00Z",
        finalized_date_central: "2026-03-09",
        calibration_fingerprint: "fingerprint_offday",
      }),
    ],
  ]);

  const cutoffs = buildSnapshotActivationCutoffs(
    [
      {
        date_central: "2026-03-05",
        pregame_cutoff_utc: "2026-03-06T00:00:00Z",
      },
    ],
    metadataById,
    "2026-03-10"
  );

  assert.deepEqual(cutoffs, [
    {
      date_central: "2026-03-05",
      pregame_cutoff_utc: "2026-03-06T00:00:00Z",
      source: "pregame",
    },
    {
      date_central: "2026-03-05",
      pregame_cutoff_utc: "2026-03-06T04:30:00Z",
      source: "end_of_day",
    },
    {
      date_central: "2026-03-09",
      pregame_cutoff_utc: "2026-03-10T03:00:00Z",
      source: "end_of_day",
    },
  ]);
});
