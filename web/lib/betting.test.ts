import assert from "node:assert/strict";
import test from "node:test";

import { normalizeBetStrategy, type BetStrategy } from "./betting-strategy";
import { computeBetDecision, computeBetDecisionsForSlate, explainBetDecision, explainBetDecisionsForSlate } from "./betting";

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

test("computeBetDecision sizes favorites with a continuous fractional-Kelly stake", () => {
  const decision = computeBetDecision({
    home_team: "SAC",
    away_team: "CHI",
    home_win_probability: 0.653,
    home_moneyline: -135,
    away_moneyline: 125,
  });

  assert.equal(decision.side, "home");
  assert.equal(decision.team, "SAC");
  assert.match(decision.reason, /underpriced after uncertainty adjustment/);
  assert.equal(decision.stake, 125);
});

test("computeBetDecision sizes underdogs with the shared value screen", () => {
  const decision = computeBetDecision({
    home_team: "BOS",
    away_team: "DAL",
    home_win_probability: 0.71,
    home_moneyline: -430,
    away_moneyline: 340,
  });

  assert.equal(decision.side, "away");
  assert.equal(decision.team, "DAL");
  assert.equal(decision.stake, 125);
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
  assert.ok(betterPrice.stake >= shorterPrice.stake);
});

test("uncertainty adjustment shrinks the raw probability toward market and peers", () => {
  const trace = explainBetDecision({
    home_team: "SAC",
    away_team: "CHI",
    home_win_probability: 0.653,
    home_moneyline: -135,
    away_moneyline: 125,
    betting_model_name: "rf",
    model_win_probabilities: {
      rf: 0.653,
      ensemble: 0.58,
      glm_logit: 0.6,
    },
  });

  assert.equal(trace.decision.side, "home");
  assert.ok((trace.candidateAdjustedProbability ?? 0) < (trace.candidateRawModelProbability ?? 1));
  assert.ok((trace.candidateAdjustedProbability ?? 0) > (trace.candidateMarketProbability ?? 0));
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
  assert.equal(decision.reason, "Conservative skips underdogs");
});

test("aggressive sizes larger than risk-adjusted on the same edge", () => {
  const riskAdjusted = buildDecision("riskAdjusted");
  const aggressive = buildDecision("aggressive");

  assert.equal(riskAdjusted.team, "NYK");
  assert.equal(aggressive.team, "NYK");
  assert.ok(aggressive.stake > riskAdjusted.stake);
});

test("legacy strategy query params normalize to the new profiles", () => {
  assert.equal(normalizeBetStrategy("balanced"), "riskAdjusted");
  assert.equal(normalizeBetStrategy("riskLoving"), "aggressive");
  assert.equal(normalizeBetStrategy("aggressiveEv"), "aggressive");
  assert.equal(normalizeBetStrategy("riskAverse"), "capitalPreservation");
});

test("explainBetDecision exposes raw and adjusted sizing steps for a bet", () => {
  const trace = explainBetDecision({
    home_team: "SAC",
    away_team: "CHI",
    home_win_probability: 0.653,
    home_moneyline: -135,
    away_moneyline: 125,
  });

  assert.equal(trace.decision.team, "SAC");
  assert.equal(trace.decision.stake, 125);
  assert.equal(trace.candidateSide, "home");
  assert.equal(trace.gates.edge, true);
  assert.equal(trace.gates.expectedValue, true);
  assert.equal(trace.gates.underdogAllowed, true);
  assert.equal(trace.gates.dailyBudget, true);
  assert.ok((trace.candidateRawModelProbability ?? 0) > (trace.candidateAdjustedProbability ?? 0));
  assert.ok((trace.kellyFraction ?? 0) > 0);
  assert.ok((trace.rawKellyUnits ?? 0) >= (trace.cappedKellyUnits ?? 0));
  assert.equal(trace.continuousStake, 125);
  assert.equal(trace.finalStake, 125);
});

test("slate-level sizing enforces the daily risk budget", () => {
  const traces = explainBetDecisionsForSlate(
    [
      {
        home_team: "SAC",
        away_team: "CHI",
        home_win_probability: 0.653,
        home_moneyline: -135,
        away_moneyline: 125,
      },
      {
        home_team: "PHI",
        away_team: "ATL",
        home_win_probability: 0.67,
        home_moneyline: -140,
        away_moneyline: 128,
      },
      {
        home_team: "NYK",
        away_team: "MIA",
        home_win_probability: 0.64,
        home_moneyline: -132,
        away_moneyline: 122,
      },
      {
        home_team: "DEN",
        away_team: "UTA",
        home_win_probability: 0.69,
        home_moneyline: -142,
        away_moneyline: 130,
      },
    ],
    "capitalPreservation",
    "continuous"
  );

  const totalRisked = traces.reduce((sum, trace) => sum + trace.finalStake, 0);
  assert.ok(totalRisked <= 250);
  assert.ok(traces.some((trace) => trace.dailyRiskCapApplied));
});

test("computeBetDecisionsForSlate returns decisions in row order", () => {
  const decisions = computeBetDecisionsForSlate(
    [
      {
        home_team: "LAL",
        away_team: "NYK",
        home_win_probability: 0.347,
        home_moneyline: 135,
        away_moneyline: -145,
      },
      {
        home_team: "SAC",
        away_team: "CHI",
        home_win_probability: 0.653,
        home_moneyline: -135,
        away_moneyline: 125,
      },
    ],
    "riskAdjusted",
    "continuous"
  );

  assert.equal(decisions.length, 2);
  assert.equal(decisions[0].team, "NYK");
  assert.equal(decisions[1].team, "SAC");
});
