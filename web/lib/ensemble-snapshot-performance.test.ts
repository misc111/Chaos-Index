import assert from "node:assert/strict";
import test from "node:test";

import { filterPerformanceTimelineSnapshots } from "./ensemble-snapshot-performance";
import type { EnsembleSnapshotRow } from "./types";

function buildSnapshot(activationDateCentral: string): EnsembleSnapshotRow {
  return {
    snapshot_key: `${activationDateCentral}::run`,
    model_name: "ensemble",
    model_run_id: `run_${activationDateCentral}`,
    ensemble_model_run_id: `run_${activationDateCentral}__ensemble`,
    finalized_at_utc: `${activationDateCentral}T18:00:00Z`,
    finalized_date_central: activationDateCentral,
    activation_date_central: activationDateCentral,
    compared_through_date_central: activationDateCentral,
    pregame_cutoff_utc: `${activationDateCentral}T23:00:00Z`,
    snapshot_id: `${activationDateCentral}-snapshot`,
    artifact_path: null,
    feature_set_version: "fset",
    calibration_fingerprint: "fingerprint",
    feature_columns: [],
    feature_count: 0,
    feature_metadata: null,
    params: null,
    metrics: null,
    tuning: null,
    selected_models: [],
    ensemble_component_columns: [],
    demoted_models: [],
    stack_base_columns: [],
    glm_feature_columns: [],
    model_feature_columns: {},
    component_models: [],
    model_commit: null,
    commit_window: [],
    replayable_games: 0,
    days_tracked: 0,
    strategies: {
      riskAdjusted: {
        total_games: 0,
        suggested_bets: 0,
        wins: 0,
        losses: 0,
        total_risked: 0,
        total_profit: 0,
        roi: 0,
        avg_edge: null,
        avg_expected_value: null,
        first_bet_date_central: null,
        last_bet_date_central: null,
      },
      aggressive: {
        total_games: 0,
        suggested_bets: 0,
        wins: 0,
        losses: 0,
        total_risked: 0,
        total_profit: 0,
        roi: 0,
        avg_edge: null,
        avg_expected_value: null,
        first_bet_date_central: null,
        last_bet_date_central: null,
      },
      capitalPreservation: {
        total_games: 0,
        suggested_bets: 0,
        wins: 0,
        losses: 0,
        total_risked: 0,
        total_profit: 0,
        roi: 0,
        avg_edge: null,
        avg_expected_value: null,
        first_bet_date_central: null,
        last_bet_date_central: null,
      },
    },
    daily: [],
    bets: [],
  };
}

test("filterPerformanceTimelineSnapshots removes the March 4 legacy snapshot from performance surfaces", () => {
  const filtered = filterPerformanceTimelineSnapshots([
    buildSnapshot("2026-03-04"),
    buildSnapshot("2026-03-05"),
    buildSnapshot("2026-03-11"),
  ]);

  assert.deepEqual(
    filtered.map((snapshot) => snapshot.activation_date_central),
    ["2026-03-05", "2026-03-11"]
  );
});
