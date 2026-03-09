import assert from "node:assert/strict";
import test from "node:test";

import { buildBetSizingExplainerModel } from "./bet-sizing-explainer";
import { buildBetSizingGamePreviews, selectBetSizingSlate } from "./bet-sizing-view";
import { getBetStrategyConfig } from "./betting-strategy";
import type { BetSizingPolicyPreview } from "./bet-sizing-view";
import type { GamesTodayResponse } from "./types";

const POLICY: BetSizingPolicyPreview = {
  key: "riskAdjusted",
  label: "Balanced",
  shortLabel: "Standard risk",
  description: "Balanced baseline.",
  matchingStrategies: ["riskAdjusted"],
  configSignature: "balanced",
  allowUnderdogs: true,
  minEdge: 0.03,
  minExpectedValue: 0.02,
  fractionalKelly: 0.5,
  maxBetUnits: 1.25,
  maxDailyUnits: 4,
  optimizationSource: "static_fallback" as const,
  metrics: null,
  frontierPoint: null,
  isFrontierPoint: false,
};

test("selectBetSizingSlate filters upcoming rows to the active Central date", () => {
  const payload: GamesTodayResponse = {
    league: "NBA",
    date_central: "2026-03-09",
    rows: [
      {
        game_id: 1,
        home_team: "CLE",
        away_team: "PHI",
        home_win_probability: 0.71,
        start_time_utc: "2026-03-09T23:00:00Z",
      },
      {
        game_id: 2,
        home_team: "BKN",
        away_team: "MEM",
        home_win_probability: 0.36,
        start_time_utc: "2026-03-10T23:30:00Z",
      },
    ],
    historical_rows: [],
  };

  const slate = selectBetSizingSlate(payload);

  assert.equal(slate.source, "upcoming");
  assert.equal(slate.rows.length, 1);
  assert.equal(slate.rows[0]?.game_id, 1);
});

test("buildBetSizingExplainerModel tracks requested versus funded stake through the daily budget", () => {
  const previews = buildBetSizingGamePreviews(
    [
      {
        game_id: 1,
        home_team: "CLE",
        away_team: "PHI",
        home_win_probability: 0.7182910267877917,
        home_moneyline: -542,
        away_moneyline: 460,
      },
      {
        game_id: 2,
        home_team: "BKN",
        away_team: "MEM",
        home_win_probability: 0.36985666583178967,
        home_moneyline: 105,
        away_moneyline: -113,
      },
      {
        game_id: 3,
        home_team: "OKC",
        away_team: "DEN",
        home_win_probability: 0.8244608655334976,
        home_moneyline: -255,
        away_moneyline: 240,
      },
    ],
    "riskAdjusted",
    POLICY
  );

  const model = buildBetSizingExplainerModel(
    previews,
    POLICY,
    {
      label: "Using today to explain the sizing flow.",
      rows: [],
      source: "upcoming",
    },
    null
  );

  assert.equal(model.totalBudget, getBetStrategyConfig("riskAdjusted").maxDailyUnits * 100);
  assert.equal(model.allocatedBudget, 375);
  assert.equal(model.remainingBudget, 25);
  assert.equal(model.fundedBetCount, 3);
  assert.equal(model.allocationSteps.length, 3);
  assert.equal(model.allocationSteps[0]?.budgetBefore, 400);
  assert.equal(model.allocationSteps[2]?.budgetAfter, 25);
  assert.equal(model.games.every((game) => game.requestedStake >= game.finalStake), true);
});
