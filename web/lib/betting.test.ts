import assert from "node:assert/strict";
import test from "node:test";

import { computeBetDecision } from "./betting";

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
