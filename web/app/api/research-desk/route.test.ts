import assert from "node:assert/strict";
import test from "node:test";

import { buildNightlyRows, buildOvernightSummary, buildUnsupportedPayload } from "./route";

test("buildNightlyRows preserves actual bet decisions from the pricing engine", () => {
  const rows = buildNightlyRows(
    {
      rows: [
        {
          game_id: 1,
          start_time_utc: "2026-03-27T23:00:00Z",
          home_team_name: "Boston Celtics",
          away_team_name: "New York Knicks",
          home_win_probability: 0.62,
          betting_model_name: "glm_ridge",
        },
        {
          game_id: 2,
          start_time_utc: "2026-03-28T01:30:00Z",
          home_team_name: "Los Angeles Lakers",
          away_team_name: "Denver Nuggets",
          home_win_probability: 0.49,
          betting_model_name: "glm_ridge",
        },
      ],
    },
    [
      {
        bet: "$75 Boston Celtics",
        reason: "Favorite underpriced after uncertainty adjustment",
        side: "home",
        team: "Boston Celtics",
        stake: 75,
        odds: -140,
        modelProbability: 0.62,
        marketProbability: 0.58,
        edge: 0.04,
        expectedValue: 0.03,
      },
      {
        bet: "$0",
        reason: "Adjusted price fair",
        side: "none",
        team: null,
        stake: 0,
        odds: null,
        modelProbability: null,
        marketProbability: null,
        edge: null,
        expectedValue: null,
      },
    ]
  );

  assert.equal(rows[0].bet_label, "bet");
  assert.equal(rows[0].team, "Boston Celtics");
  assert.equal(rows[0].stake, 75);
  assert.equal(rows[1].bet_label, "pass");
  assert.equal(rows[1].reason, "Adjusted price fair");
});

test("buildOvernightSummary mentions posture, promotion outcome, and slate counts", () => {
  const summary = buildOvernightSummary({
    deskPosture: "guarded",
    championModelName: "glm_ridge",
    promotion: {
      promoted: false,
      incumbent_model_name: "glm_ridge",
      candidate_model_name: "glm_lasso",
      reason_summary: "Rejected: calibration_guardrail, minimum_profitable_folds",
      policy: null,
      created_at_utc: "2026-03-27T12:00:00Z",
    },
    totalGames: 4,
    betCount: 1,
  });

  assert.match(summary, /Guarded posture is active/i);
  assert.match(summary, /Active champion: glm_ridge/i);
  assert.match(summary, /calibration_guardrail/i);
  assert.match(summary, /4 games/i);
  assert.match(summary, /1 bet/i);
});

test("buildUnsupportedPayload returns the NBA-first fallback state", () => {
  const payload = buildUnsupportedPayload("NHL");

  assert.equal(payload.league, "NHL");
  assert.equal(payload.rows.length, 0);
  assert.equal(payload.counts.total_games, 0);
  assert.match(String(payload.overnight_summary), /NBA only/i);
});
