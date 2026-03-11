import assert from "node:assert/strict";
import test from "node:test";

import { HISTORICAL_BANKROLL_START_DOLLARS } from "./betting";
import { buildEnsembleSnapshotBankrollSeries, listEnsembleSnapshotChartDates } from "./ensemble-snapshot-chart";
import type { EnsembleSnapshotRow } from "./types";

function buildSnapshot(overrides: Partial<EnsembleSnapshotRow>): EnsembleSnapshotRow {
  return {
    snapshot_key: "snapshot_default",
    model_name: "ensemble",
    model_run_id: "run_default",
    ensemble_model_run_id: "run_default__ensemble",
    finalized_at_utc: "2026-03-05T18:00:00Z",
    finalized_date_central: "2026-03-05",
    activation_date_central: "2026-03-05",
    compared_through_date_central: "2026-03-06",
    pregame_cutoff_utc: "2026-03-05T23:00:00Z",
    snapshot_id: "snapshot-id",
    artifact_path: "artifacts/validation/nba/run_default",
    feature_set_version: "fset_default",
    calibration_fingerprint: "fingerprint_default",
    feature_columns: ["rest_diff"],
    feature_count: 1,
    feature_metadata: null,
    params: null,
    metrics: null,
    tuning: null,
    selected_models: ["rf"],
    ensemble_component_columns: ["rf"],
    demoted_models: [],
    stack_base_columns: ["rf"],
    glm_feature_columns: [],
    model_feature_columns: null,
    component_models: [],
    model_commit: null,
    commit_window: [],
    replayable_games: 2,
    days_tracked: 2,
    strategies: {
      riskAdjusted: {
        total_games: 2,
        suggested_bets: 2,
        wins: 1,
        losses: 1,
        total_risked: 240,
        total_profit: 80,
        roi: 80 / 240,
        avg_edge: 0.05,
        avg_expected_value: 0.03,
        first_bet_date_central: "2026-03-05",
        last_bet_date_central: "2026-03-06",
      },
      aggressive: {
        total_games: 2,
        suggested_bets: 2,
        wins: 1,
        losses: 1,
        total_risked: 360,
        total_profit: 120,
        roi: 120 / 360,
        avg_edge: 0.05,
        avg_expected_value: 0.03,
        first_bet_date_central: "2026-03-05",
        last_bet_date_central: "2026-03-06",
      },
      capitalPreservation: {
        total_games: 2,
        suggested_bets: 1,
        wins: 1,
        losses: 0,
        total_risked: 90,
        total_profit: 90,
        roi: 1,
        avg_edge: 0.05,
        avg_expected_value: 0.03,
        first_bet_date_central: "2026-03-05",
        last_bet_date_central: "2026-03-05",
      },
    },
    daily: [
      {
        date_central: "2026-03-05",
        slate_games: 3,
        strategies: {
          riskAdjusted: {
            slate_games: 3,
            suggested_bets: 1,
            wins: 1,
            losses: 0,
            total_risked: 120,
            total_profit: 120,
            cumulative_risked: 120,
            cumulative_profit: 120,
            roi: 1,
            cumulative_roi: 1,
          },
          aggressive: {
            slate_games: 3,
            suggested_bets: 1,
            wins: 1,
            losses: 0,
            total_risked: 180,
            total_profit: 180,
            cumulative_risked: 180,
            cumulative_profit: 180,
            roi: 1,
            cumulative_roi: 1,
          },
          capitalPreservation: {
            slate_games: 3,
            suggested_bets: 1,
            wins: 1,
            losses: 0,
            total_risked: 90,
            total_profit: 90,
            cumulative_risked: 90,
            cumulative_profit: 90,
            roi: 1,
            cumulative_roi: 1,
          },
        },
      },
      {
        date_central: "2026-03-06",
        slate_games: 4,
        strategies: {
          riskAdjusted: {
            slate_games: 4,
            suggested_bets: 1,
            wins: 0,
            losses: 1,
            total_risked: 120,
            total_profit: -40,
            cumulative_risked: 240,
            cumulative_profit: 80,
            roi: -40 / 120,
            cumulative_roi: 80 / 240,
          },
          aggressive: {
            slate_games: 4,
            suggested_bets: 1,
            wins: 0,
            losses: 1,
            total_risked: 180,
            total_profit: -60,
            cumulative_risked: 360,
            cumulative_profit: 120,
            roi: -60 / 180,
            cumulative_roi: 120 / 360,
          },
          capitalPreservation: {
            slate_games: 4,
            suggested_bets: 0,
            wins: 0,
            losses: 0,
            total_risked: 0,
            total_profit: 0,
            cumulative_risked: 90,
            cumulative_profit: 90,
            roi: 0,
            cumulative_roi: 1,
          },
        },
      },
    ],
    bets: [],
    ...overrides,
  };
}

test("buildEnsembleSnapshotBankrollSeries anchors same-day snapshots to the prior day and converts cumulative profit into bankroll", () => {
  const series = buildEnsembleSnapshotBankrollSeries([buildSnapshot({})], "riskAdjusted");
  const conservativeSeries = buildEnsembleSnapshotBankrollSeries([buildSnapshot({})], "capitalPreservation");

  assert.equal(series.length, 1);
  assert.equal(series[0].points[0].date_central, "2026-03-04");
  assert.equal(series[0].points[0].cumulative_bankroll, HISTORICAL_BANKROLL_START_DOLLARS);
  assert.equal(series[0].points[1].cumulative_bankroll, HISTORICAL_BANKROLL_START_DOLLARS + 120);
  assert.equal(series[0].points[2].cumulative_bankroll, HISTORICAL_BANKROLL_START_DOLLARS + 80);
  assert.equal(series[0].final_point.cumulative_profit, 80);
  assert.equal(conservativeSeries[0].final_point.cumulative_profit, 90);
});

test("buildEnsembleSnapshotBankrollSeries keeps pending snapshots on their activation date when no settled games exist", () => {
  const series = buildEnsembleSnapshotBankrollSeries(
    [
      buildSnapshot({
        snapshot_key: "snapshot_pending",
        activation_date_central: "2026-03-11",
        compared_through_date_central: null,
        replayable_games: 0,
        days_tracked: 0,
        daily: [],
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
      }),
    ],
    "riskAdjusted"
  );

  assert.equal(series[0].points.length, 1);
  assert.equal(series[0].points[0].kind, "pending");
  assert.equal(series[0].points[0].date_central, "2026-03-11");
});

test("listEnsembleSnapshotChartDates returns a sorted union of every plotted bankroll date", () => {
  const series = buildEnsembleSnapshotBankrollSeries(
    [
      buildSnapshot({ snapshot_key: "snapshot_a", activation_date_central: "2026-03-05" }),
      buildSnapshot({
        snapshot_key: "snapshot_b",
        activation_date_central: "2026-03-07",
        daily: [
          {
            date_central: "2026-03-08",
            slate_games: 2,
            strategies: {
              riskAdjusted: {
                slate_games: 2,
                suggested_bets: 1,
                wins: 1,
                losses: 0,
                total_risked: 100,
                total_profit: 90,
                cumulative_risked: 100,
                cumulative_profit: 90,
                roi: 0.9,
                cumulative_roi: 0.9,
              },
              aggressive: {
                slate_games: 2,
                suggested_bets: 1,
                wins: 1,
                losses: 0,
                total_risked: 150,
                total_profit: 135,
                cumulative_risked: 150,
                cumulative_profit: 135,
                roi: 0.9,
                cumulative_roi: 0.9,
              },
              capitalPreservation: {
                slate_games: 2,
                suggested_bets: 1,
                wins: 1,
                losses: 0,
                total_risked: 75,
                total_profit: 70,
                cumulative_risked: 75,
                cumulative_profit: 70,
                roi: 70 / 75,
                cumulative_roi: 70 / 75,
              },
            },
          },
        ],
      }),
    ],
    "riskAdjusted"
  );

  assert.deepEqual(listEnsembleSnapshotChartDates(series), ["2026-03-04", "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08"]);
});
