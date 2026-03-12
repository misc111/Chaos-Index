import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPerformanceReplayExperimentToEnsembleSnapshots,
  buildPerformanceExperimentStagingFileName,
  defaultPerformanceReplayExperimentForLeague,
  normalizePerformanceReplayExperiment,
} from "./performance-replay-experiments";
import type { EnsembleSnapshotRow } from "./types";

function buildStrategyDetail(stake: number, odds: number, outcome: "win" | "loss" | "no_bet", profit: number) {
  return {
    bet_label: stake > 0 ? `$${stake}` : "$0",
    reason: "Baseline replay decision",
    side: odds > 0 ? "away" : "home",
    team: odds > 0 ? "DOG" : "FAV",
    stake,
    odds,
    model_probability: 0.61,
    market_probability: 0.52,
    edge: 0.09,
    expected_value: 0.18,
    outcome,
    profit,
    payout: outcome === "win" ? stake + profit : 0,
  } as const;
}

function buildSnapshot(): EnsembleSnapshotRow {
  return {
    snapshot_key: "2026-03-05::run_demo",
    model_name: "ensemble",
    model_run_id: "run_demo",
    ensemble_model_run_id: "run_demo__ensemble",
    finalized_at_utc: "2026-03-05T15:00:00Z",
    finalized_date_central: "2026-03-05",
    activation_date_central: "2026-03-05",
    compared_through_date_central: "2026-03-08",
    pregame_cutoff_utc: "2026-03-08T23:00:00Z",
    snapshot_id: "snap_demo",
    artifact_path: null,
    feature_set_version: "fset_demo",
    calibration_fingerprint: "fingerprint_demo",
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
    replayable_games: 3,
    days_tracked: 3,
    strategies: {
      riskAdjusted: {
        total_games: 3,
        suggested_bets: 3,
        wins: 1,
        losses: 2,
        total_risked: 300,
        total_profit: -25,
        roi: -25 / 300,
        avg_edge: 0.09,
        avg_expected_value: 0.18,
        first_bet_date_central: "2026-03-05",
        last_bet_date_central: "2026-03-08",
      },
      aggressive: {
        total_games: 3,
        suggested_bets: 3,
        wins: 1,
        losses: 2,
        total_risked: 450,
        total_profit: -50,
        roi: -50 / 450,
        avg_edge: 0.09,
        avg_expected_value: 0.18,
        first_bet_date_central: "2026-03-05",
        last_bet_date_central: "2026-03-08",
      },
      capitalPreservation: {
        total_games: 3,
        suggested_bets: 1,
        wins: 1,
        losses: 0,
        total_risked: 75,
        total_profit: 57.69,
        roi: 57.69 / 75,
        avg_edge: 0.09,
        avg_expected_value: 0.18,
        first_bet_date_central: "2026-03-05",
        last_bet_date_central: "2026-03-05",
      },
    },
    daily: [
      {
        date_central: "2026-03-05",
        slate_games: 1,
        strategies: {
          riskAdjusted: {
            slate_games: 1,
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
            slate_games: 1,
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
            slate_games: 1,
            suggested_bets: 1,
            wins: 1,
            losses: 0,
            total_risked: 75,
            total_profit: 57.69,
            cumulative_risked: 75,
            cumulative_profit: 57.69,
            roi: 0.7692,
            cumulative_roi: 0.7692,
          },
        },
      },
      {
        date_central: "2026-03-06",
        slate_games: 1,
        strategies: {
          riskAdjusted: {
            slate_games: 1,
            suggested_bets: 1,
            wins: 0,
            losses: 1,
            total_risked: 100,
            total_profit: -100,
            cumulative_risked: 200,
            cumulative_profit: -10,
            roi: -1,
            cumulative_roi: -0.05,
          },
          aggressive: {
            slate_games: 1,
            suggested_bets: 1,
            wins: 0,
            losses: 1,
            total_risked: 150,
            total_profit: -150,
            cumulative_risked: 300,
            cumulative_profit: -15,
            roi: -1,
            cumulative_roi: -0.05,
          },
          capitalPreservation: {
            slate_games: 1,
            suggested_bets: 0,
            wins: 0,
            losses: 0,
            total_risked: 0,
            total_profit: 0,
            cumulative_risked: 75,
            cumulative_profit: 57.69,
            roi: 0,
            cumulative_roi: 0.7692,
          },
        },
      },
      {
        date_central: "2026-03-08",
        slate_games: 1,
        strategies: {
          riskAdjusted: {
            slate_games: 1,
            suggested_bets: 1,
            wins: 0,
            losses: 1,
            total_risked: 100,
            total_profit: -15,
            cumulative_risked: 300,
            cumulative_profit: -25,
            roi: -0.15,
            cumulative_roi: -0.0833,
          },
          aggressive: {
            slate_games: 1,
            suggested_bets: 1,
            wins: 0,
            losses: 1,
            total_risked: 150,
            total_profit: -35,
            cumulative_risked: 450,
            cumulative_profit: -50,
            roi: -0.2333,
            cumulative_roi: -0.1111,
          },
          capitalPreservation: {
            slate_games: 1,
            suggested_bets: 0,
            wins: 0,
            losses: 0,
            total_risked: 0,
            total_profit: 0,
            cumulative_risked: 75,
            cumulative_profit: 57.69,
            roi: 0,
            cumulative_roi: 0.7692,
          },
        },
      },
    ],
    bets: [
      {
        game_id: 1,
        date_central: "2026-03-05",
        forecast_as_of_utc: "2026-03-05T15:00:00Z",
        start_time_utc: "2026-03-05T23:30:00Z",
        final_utc: "2026-03-06T02:00:00Z",
        home_team: "A",
        away_team: "B",
        home_score: 110,
        away_score: 100,
        home_moneyline: -130,
        away_moneyline: 120,
        strategies: {
          riskAdjusted: buildStrategyDetail(100, -130, "win", 90),
          aggressive: buildStrategyDetail(150, -130, "win", 135),
          capitalPreservation: buildStrategyDetail(75, -130, "win", 57.69),
        },
      },
      {
        game_id: 2,
        date_central: "2026-03-06",
        forecast_as_of_utc: "2026-03-05T15:00:00Z",
        start_time_utc: "2026-03-06T23:30:00Z",
        final_utc: "2026-03-07T02:00:00Z",
        home_team: "C",
        away_team: "D",
        home_score: 90,
        away_score: 101,
        home_moneyline: 450,
        away_moneyline: -500,
        strategies: {
          riskAdjusted: buildStrategyDetail(100, 450, "loss", -100),
          aggressive: buildStrategyDetail(150, 450, "loss", -150),
          capitalPreservation: buildStrategyDetail(0, 450, "no_bet", 0),
        },
      },
      {
        game_id: 3,
        date_central: "2026-03-08",
        forecast_as_of_utc: "2026-03-05T15:00:00Z",
        start_time_utc: "2026-03-08T23:30:00Z",
        final_utc: "2026-03-09T02:00:00Z",
        home_team: "E",
        away_team: "F",
        home_score: 99,
        away_score: 96,
        home_moneyline: -110,
        away_moneyline: 100,
        strategies: {
          riskAdjusted: buildStrategyDetail(100, -110, "loss", -15),
          aggressive: buildStrategyDetail(150, -110, "loss", -35),
          capitalPreservation: buildStrategyDetail(0, -110, "no_bet", 0),
        },
      },
    ],
  };
}

test("normalizePerformanceReplayExperiment accepts only known experiment ids", () => {
  assert.equal(normalizePerformanceReplayExperiment("fresh-1d-no-dogs-over-300"), "fresh-1d-no-dogs-over-300");
  assert.equal(normalizePerformanceReplayExperiment("unknown"), null);
  assert.equal(buildPerformanceExperimentStagingFileName("fresh-1d-no-dogs-over-300"), "performance.fresh-1d-no-dogs-over-300.json");
  assert.equal(buildPerformanceExperimentStagingFileName("unknown"), "performance.json");
  assert.equal(defaultPerformanceReplayExperimentForLeague("NBA"), "fresh-1d-no-dogs-over-300");
  assert.equal(defaultPerformanceReplayExperimentForLeague("NHL"), null);
});

test("applyPerformanceReplayExperimentToEnsembleSnapshots zeroes stale bets and giant dogs but keeps the snapshot path intact", () => {
  const transformed = applyPerformanceReplayExperimentToEnsembleSnapshots(
    [buildSnapshot()],
    "fresh-1d-no-dogs-over-300"
  );

  assert.equal(transformed.length, 1);
  assert.equal(transformed[0].replayable_games, 3);
  assert.equal(transformed[0].strategies.aggressive.suggested_bets, 1);
  assert.equal(transformed[0].strategies.aggressive.total_risked, 150);
  assert.equal(transformed[0].strategies.aggressive.total_profit, 135);
  assert.equal(transformed[0].daily[0].strategies.aggressive.total_profit, 135);
  assert.equal(transformed[0].daily[1].strategies.aggressive.total_profit, 0);
  assert.equal(transformed[0].daily[2].strategies.aggressive.total_profit, 0);
  assert.equal(transformed[0].bets[1].strategies.aggressive.stake, 0);
  assert.match(transformed[0].bets[1].strategies.aggressive.reason, /underdogs above \+300/);
  assert.equal(transformed[0].bets[2].strategies.aggressive.stake, 0);
  assert.match(transformed[0].bets[2].strategies.aggressive.reason, /more than 1 day after the snapshot date/);
});
