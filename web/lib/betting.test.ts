import assert from "node:assert/strict";
import test from "node:test";

import { normalizeBetStrategy, type BetSizingStyle, type BetStrategy } from "./betting-strategy";
import { computeBetDecision } from "./betting";

function buildDecision(strategy?: BetStrategy, sizingStyle?: BetSizingStyle) {
  return computeBetDecision(
    {
      home_team: "LAL",
      away_team: "NYK",
      home_win_probability: 0.347,
      home_moneyline: 135,
      away_moneyline: -145,
    },
    strategy,
    sizingStyle
  );
}

test("computeBetDecision sizes favorites with a continuous Kelly stake", () => {
  const decision = computeBetDecision({
    home_team: "SAC",
    away_team: "CHI",
    home_win_probability: 0.653,
    home_moneyline: -135,
    away_moneyline: 125,
  });

  assert.equal(decision.side, "home");
  assert.equal(decision.team, "SAC");
  assert.equal(decision.reason, "Favorite underpriced");
  assert.equal(decision.stake, 90);
});

test("computeBetDecision can snap favorite stakes into legacy buckets", () => {
  const decision = computeBetDecision(
    {
      home_team: "SAC",
      away_team: "CHI",
      home_win_probability: 0.653,
      home_moneyline: -135,
      away_moneyline: 125,
    },
    "riskAdjusted",
    "bucketed"
  );

  assert.equal(decision.side, "home");
  assert.equal(decision.team, "SAC");
  assert.equal(decision.reason, "Favorite underpriced");
  assert.equal(decision.stake, 100);
});

test("computeBetDecision sizes underdogs with a continuous Kelly stake", () => {
  const decision = computeBetDecision({
    home_team: "BOS",
    away_team: "DAL",
    home_win_probability: 0.71,
    home_moneyline: -430,
    away_moneyline: 340,
  });

  assert.equal(decision.side, "away");
  assert.equal(decision.team, "DAL");
  assert.equal(decision.reason, "Underdog underpriced");
  assert.equal(decision.stake, 40);
});

test("computeBetDecision can snap underdog stakes into legacy buckets", () => {
  const decision = computeBetDecision(
    {
      home_team: "BOS",
      away_team: "DAL",
      home_win_probability: 0.71,
      home_moneyline: -430,
      away_moneyline: 340,
    },
    "riskAdjusted",
    "bucketed"
  );

  assert.equal(decision.side, "away");
  assert.equal(decision.team, "DAL");
  assert.equal(decision.reason, "Underdog underpriced");
  assert.equal(decision.stake, 50);
});

test("computeBetDecision increases stake when the same side gets a better price", () => {
  const shorterPrice = computeBetDecision({
    home_team: "NYK",
    away_team: "LAL",
    home_win_probability: 0.653,
    home_moneyline: -145,
    away_moneyline: 135,
  });
  const betterPrice = computeBetDecision({
    home_team: "NYK",
    away_team: "LAL",
    home_win_probability: 0.653,
    home_moneyline: -125,
    away_moneyline: 115,
  });

  assert.equal(shorterPrice.side, "home");
  assert.equal(betterPrice.side, "home");
  assert.ok(betterPrice.stake > shorterPrice.stake);
});

test("capital preservation skips underdogs entirely", () => {
  const decision = computeBetDecision(
    {
      home_team: "NO",
      away_team: "WSH",
      home_win_probability: 0.61,
      home_moneyline: -430,
      away_moneyline: 360,
    },
    "capitalPreservation"
  );

  assert.equal(decision.side, "none");
  assert.equal(decision.stake, 0);
  assert.equal(decision.reason, "Capital Preservation skips underdogs");
});

test("aggressive sizes larger than risk-adjusted on the same edge", () => {
  const riskAdjusted = buildDecision("riskAdjusted");
  const aggressive = buildDecision("aggressive");

  assert.equal(riskAdjusted.team, "NYK");
  assert.equal(aggressive.team, "NYK");
  assert.ok(aggressive.stake > riskAdjusted.stake);
});

test("bucketed sizing preserves profile differences through bucket selection", () => {
  const riskAdjusted = buildDecision("riskAdjusted", "bucketed");
  const aggressive = buildDecision("aggressive", "bucketed");

  assert.equal(riskAdjusted.team, "NYK");
  assert.equal(aggressive.team, "NYK");
  assert.ok(aggressive.stake >= riskAdjusted.stake);
});

test("legacy strategy query params normalize to the new profiles", () => {
  assert.equal(normalizeBetStrategy("balanced"), "riskAdjusted");
  assert.equal(normalizeBetStrategy("riskLoving"), "aggressive");
  assert.equal(normalizeBetStrategy("aggressiveEv"), "aggressive");
  assert.equal(normalizeBetStrategy("riskAverse"), "capitalPreservation");
});
