import assert from "node:assert/strict";
import test from "node:test";

import type { BetStrategy } from "./betting-strategy";
import { computeBetDecision } from "./betting";

function buildDecision(strategy?: BetStrategy) {
  return computeBetDecision(
    {
      home_team: "LAL",
      away_team: "NYK",
      home_win_probability: 0.347,
      home_moneyline: 135,
      away_moneyline: -145,
    },
    strategy
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
  assert.equal(decision.stake, 125);
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
  assert.equal(decision.stake, 55);
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

test("risk-averse strategy skips underdogs entirely", () => {
  const decision = computeBetDecision(
    {
      home_team: "NO",
      away_team: "WSH",
      home_win_probability: 0.61,
      home_moneyline: -430,
      away_moneyline: 360,
    },
    "riskAverse"
  );

  assert.equal(decision.side, "none");
  assert.equal(decision.stake, 0);
  assert.equal(decision.reason, "Risk-averse profile skips underdogs");
});

test("risk-loving strategy sizes larger than balanced on the same edge", () => {
  const balanced = buildDecision("balanced");
  const riskLoving = buildDecision("riskLoving");

  assert.equal(balanced.team, "NYK");
  assert.equal(riskLoving.team, "NYK");
  assert.ok(riskLoving.stake > balanced.stake);
});
