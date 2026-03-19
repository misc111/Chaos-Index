import assert from "node:assert/strict";
import test from "node:test";

import { buildEnsembleSnapshots } from "./ensemble-snapshot-replay";
import type { EnsembleSnapshotRunMetadata } from "./ensemble-snapshot-replay";

test("buildEnsembleSnapshots replays frozen snapshots only from activation day forward and keeps conservative results available", () => {
  const snapshots = buildEnsembleSnapshots(
    [
      {
        game_id: 10,
        date_central: "2026-03-04",
        start_time_utc: "2026-03-05T00:00:00Z",
        final_utc: "2026-03-05T03:00:00Z",
        home_team: "NYK",
        away_team: "MIA",
        home_score: 99,
        away_score: 94,
        home_win: 1,
        forecast_as_of_utc: "2026-03-04T20:00:00Z",
        home_moneyline: -130,
        away_moneyline: 120,
        model_run_id: "run_thu",
        home_win_probability: 0.66,
        model_win_probabilities: {
          ensemble: 0.66,
          rf: 0.62,
          glm_ridge: 0.61,
        },
      },
      {
        game_id: 11,
        date_central: "2026-03-05",
        start_time_utc: "2026-03-06T00:00:00Z",
        final_utc: "2026-03-06T03:00:00Z",
        home_team: "SAC",
        away_team: "CHI",
        home_score: 112,
        away_score: 101,
        home_win: 1,
        forecast_as_of_utc: "2026-03-05T20:00:00Z",
        home_moneyline: -135,
        away_moneyline: 125,
        model_run_id: "run_thu",
        home_win_probability: 0.653,
        model_win_probabilities: {
          ensemble: 0.653,
          rf: 0.612,
          glm_ridge: 0.601,
        },
      },
      {
        game_id: 12,
        date_central: "2026-03-06",
        start_time_utc: "2026-03-07T00:00:00Z",
        final_utc: "2026-03-07T03:00:00Z",
        home_team: "DEN",
        away_team: "UTA",
        home_score: 119,
        away_score: 108,
        home_win: 1,
        forecast_as_of_utc: "2026-03-06T20:00:00Z",
        home_moneyline: -142,
        away_moneyline: 130,
        model_run_id: "run_thu",
        home_win_probability: 0.69,
        model_win_probabilities: {
          ensemble: 0.69,
          rf: 0.651,
          glm_ridge: 0.633,
        },
      },
      {
        game_id: 13,
        date_central: "2026-03-07",
        start_time_utc: "2026-03-08T00:00:00Z",
        final_utc: "2026-03-08T03:00:00Z",
        home_team: "BOS",
        away_team: "CLE",
        home_score: 103,
        away_score: 108,
        home_win: 0,
        forecast_as_of_utc: "2026-03-07T16:00:00Z",
        home_moneyline: -118,
        away_moneyline: 108,
        model_run_id: "run_sat",
        home_win_probability: 0.42,
        model_win_probabilities: {
          ensemble: 0.42,
          rf: 0.46,
          glm_ridge: 0.44,
        },
      },
    ],
    [
      {
        activation_date_central: "2026-03-05",
        pregame_cutoff_utc: "2026-03-06T00:00:00Z",
        model_run_id: "run_thu",
      },
      {
        activation_date_central: "2026-03-07",
        pregame_cutoff_utc: "2026-03-08T00:00:00Z",
        model_run_id: "run_sat",
      },
    ],
    new Map<string, EnsembleSnapshotRunMetadata>([
      [
        "run_thu",
        {
          model_name: "ensemble",
          model_run_id: "run_thu",
          ensemble_model_run_id: "run_thu__ensemble",
          finalized_at_utc: "2026-03-05T22:20:41Z",
          finalized_date_central: "2026-03-05",
          feature_set_version: "fset_old",
          calibration_fingerprint: "fingerprint_old",
          feature_columns: ["rest_diff", "elo_home_prob"],
          selected_models: ["elo_baseline", "rf", "glm_ridge"],
          ensemble_component_columns: ["elo_baseline", "rf", "glm_ridge"],
          demoted_models: [],
          stack_base_columns: ["elo_baseline", "rf", "glm_ridge"],
          glm_feature_columns: ["rest_diff", "elo_home_prob"],
          model_feature_columns: { rf: ["rest_diff"], glm_ridge: ["rest_diff", "elo_home_prob"] },
          component_models: [],
          model_commit: {
            sha: "abc1234abc1234",
            short_sha: "abc1234",
            committed_at_utc: "2026-03-05T18:00:00Z",
            subject: "Calibrate Thursday ensemble",
          },
          commit_window: [],
        },
      ],
      [
        "run_sat",
        {
          model_name: "ensemble",
          model_run_id: "run_sat",
          ensemble_model_run_id: "run_sat__ensemble",
          finalized_at_utc: "2026-03-07T18:00:00Z",
          finalized_date_central: "2026-03-07",
          feature_set_version: "fset_new",
          calibration_fingerprint: "fingerprint_new",
          feature_columns: ["rest_diff", "elo_home_prob", "injury_rotation"],
          selected_models: ["elo_baseline", "rf", "glm_ridge", "glm_elastic_net"],
          ensemble_component_columns: ["elo_baseline", "rf", "glm_ridge", "glm_elastic_net"],
          demoted_models: [],
          stack_base_columns: ["elo_baseline", "rf", "glm_ridge", "glm_elastic_net"],
          glm_feature_columns: ["rest_diff", "elo_home_prob", "injury_rotation"],
          model_feature_columns: {
            rf: ["rest_diff"],
            glm_ridge: ["rest_diff", "elo_home_prob"],
            glm_elastic_net: ["rest_diff", "elo_home_prob", "injury_rotation"],
          },
          component_models: [],
          model_commit: {
            sha: "def5678def5678",
            short_sha: "def5678",
            committed_at_utc: "2026-03-07T16:00:00Z",
            subject: "Ship Saturday recalibration",
          },
          commit_window: [],
        },
      ],
    ]),
    "NBA"
  );

  assert.equal(snapshots.length, 2);
  assert.equal(snapshots[0].model_run_id, "run_sat");
  assert.equal(snapshots[1].model_run_id, "run_thu");
  assert.equal(snapshots[1].bets.length, 2);
  assert.equal(snapshots[1].daily.length, 2);
  assert.equal(snapshots[1].daily[0].date_central, "2026-03-05");
  assert.equal(
    snapshots[1].daily[1].strategies.riskAdjusted.cumulative_profit,
    snapshots[1].strategies.riskAdjusted.total_profit
  );
  assert.ok(snapshots[1].daily[1].strategies.capitalPreservation.cumulative_profit > 0);
  assert.ok(
    snapshots[1].strategies.capitalPreservation.total_risked <= snapshots[1].strategies.riskAdjusted.total_risked
  );
  assert.ok(
    snapshots[1].bets[0].strategies.capitalPreservation.stake <
      snapshots[1].bets[0].strategies.aggressive.stake
  );
  assert.equal(snapshots[1].feature_count, 2);
  assert.equal(snapshots[0].activation_date_central, "2026-03-07");
  assert.ok(snapshots[1].strategies.aggressive.total_risked > snapshots[1].strategies.riskAdjusted.total_risked);
});
