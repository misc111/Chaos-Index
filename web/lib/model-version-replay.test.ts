import assert from "node:assert/strict";
import test from "node:test";

import { buildModelReplayRuns } from "./model-version-replay";
import { getBetStrategyConfig } from "./betting-strategy";
import type { ModelReplayRunMetadata } from "./model-version-replay";

test("buildModelReplayRuns groups dated model snapshots and preserves all supported bet objectives", () => {
  const runs = buildModelReplayRuns(
    [
      {
        game_id: 1,
        date_central: "2026-03-06",
        start_time_utc: "2026-03-06T01:00:00Z",
        final_utc: "2026-03-06T04:00:00Z",
        home_team: "SAC",
        away_team: "CHI",
        home_score: 112,
        away_score: 101,
        home_win: 1,
        forecast_as_of_utc: "2026-03-06T00:00:00Z",
        home_moneyline: -135,
        away_moneyline: 125,
        model_name: "ensemble",
        model_run_id: "run_old",
        home_win_probability: 0.653,
        model_win_probabilities: {
          ensemble: 0.653,
          rf: 0.612,
          glm_ridge: 0.601,
        },
      },
      {
        game_id: 2,
        date_central: "2026-03-07",
        start_time_utc: "2026-03-07T01:00:00Z",
        final_utc: "2026-03-07T04:00:00Z",
        home_team: "DEN",
        away_team: "UTA",
        home_score: 119,
        away_score: 108,
        home_win: 1,
        forecast_as_of_utc: "2026-03-07T00:00:00Z",
        home_moneyline: -142,
        away_moneyline: 130,
        model_name: "ensemble",
        model_run_id: "run_new",
        home_win_probability: 0.69,
        model_win_probabilities: {
          ensemble: 0.69,
          rf: 0.651,
          glm_ridge: 0.633,
        },
      },
    ],
    new Map<string, ModelReplayRunMetadata>([
      [
        "run_old",
        {
          model_name: "ensemble",
          model_run_id: "run_old",
          created_at_utc: "2026-03-06T00:10:00Z",
          feature_set_version: "fset_old",
          feature_columns: ["rest_diff", "elo_home_prob"],
          params: { weights: { ensemble: 1 } },
        },
      ],
      [
        "run_new",
        {
          model_name: "ensemble",
          model_run_id: "run_new",
          created_at_utc: "2026-03-07T00:10:00Z",
          feature_set_version: "fset_new",
          feature_columns: ["rest_diff", "elo_home_prob", "rotation_stability"],
          params: { weights: { ensemble: 1 } },
        },
      ],
    ]),
    new Map([
      [
        "run_old",
        {
          model_name: "ensemble",
          model_run_id: "run_old",
          n_games: 8,
          avg_log_loss: 0.61,
          avg_brier: 0.21,
          accuracy: 0.75,
          version_rank: 2,
          is_latest_version: 0,
        },
      ],
      [
        "run_new",
        {
          model_name: "ensemble",
          model_run_id: "run_new",
          n_games: 6,
          avg_log_loss: 0.57,
          avg_brier: 0.19,
          accuracy: 0.83,
          version_rank: 1,
          is_latest_version: 1,
        },
      ],
    ]),
    "NBA"
  );

  assert.equal(runs.length, 2);
  assert.equal(runs[0].model_run_id, "run_new");
  assert.equal(runs[0].feature_count, 3);
  assert.equal(runs[0].strategies.riskAdjusted.suggested_bets, 1);
  assert.equal(runs[0].bets[0].strategies.riskAdjusted.outcome, "win");
  assert.ok(runs[0].strategies.aggressive.total_risked > runs[0].strategies.riskAdjusted.total_risked);
  assert.ok(runs[0].bets[0].strategies.capitalPreservation.stake > 0);
  assert.ok(
    runs[0].bets[0].strategies.capitalPreservation.stake < runs[0].bets[0].strategies.aggressive.stake
  );
  assert.ok(
    runs[0].strategies.capitalPreservation.total_risked <= runs[0].strategies.aggressive.total_risked,
    getBetStrategyConfig("capitalPreservation").description
  );
  assert.equal(runs[1].feature_set_version, "fset_old");
  assert.equal(runs[1].scored_games, 8);
});
