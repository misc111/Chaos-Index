import assert from "node:assert/strict";
import test from "node:test";

import {
  getBetStrategyConfig,
  getDefaultBetStrategyForLeague,
  normalizeBetStrategy,
  type BetStrategy,
} from "./betting-strategy";
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

test("computeBetDecision sizes favorites with the bankroll-linked formula", () => {
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

test("NBA long-shot guardrail blocks extreme underdogs while NHL keeps baseline behavior", () => {
  const baseRow = {
    home_team: "BOS",
    away_team: "DAL",
    home_win_probability: 0.71,
    home_moneyline: -430,
    away_moneyline: 340,
  };

  const nbaDecision = computeBetDecision({ ...baseRow, league: "NBA" }, "riskAdjusted");
  const nhlDecision = computeBetDecision({ ...baseRow, league: "NHL" }, "riskAdjusted");

  assert.equal(nbaDecision.side, "none");
  assert.equal(nbaDecision.stake, 0);
  assert.match(nbaDecision.reason, /long-shot underdogs/i);
  assert.equal(nhlDecision.side, "away");
  assert.ok(nhlDecision.stake > 0);
});

test("guarded risk regime throttles NBA stakes and does not alter NHL stakes", () => {
  const nbaRow = {
    league: "NBA" as const,
    home_team: "SAC",
    away_team: "CHI",
    home_win_probability: 0.653,
    home_moneyline: -135,
    away_moneyline: 125,
  };
  const nhlRow = {
    ...nbaRow,
    league: "NHL" as const,
  };

  const nbaNormal = computeBetDecision(nbaRow, "riskAdjusted", undefined, { league: "NBA", riskRegime: "normal" });
  const nbaGuarded = computeBetDecision(nbaRow, "riskAdjusted", undefined, { league: "NBA", riskRegime: "guarded" });
  const nhlNormal = computeBetDecision(nhlRow, "riskAdjusted", undefined, { league: "NHL", riskRegime: "normal" });
  const nhlGuarded = computeBetDecision(nhlRow, "riskAdjusted", undefined, { league: "NHL", riskRegime: "guarded" });

  assert.ok(nbaNormal.stake > 0);
  assert.ok(nbaGuarded.stake <= nbaNormal.stake);
  assert.equal(nhlGuarded.stake, nhlNormal.stake);
});

test("NBA risk-adjusted skips long-shot underdogs above +300", () => {
  const decision = computeBetDecision({
    league: "NBA",
    home_team: "BOS",
    away_team: "DAL",
    home_win_probability: 0.71,
    home_moneyline: -430,
    away_moneyline: 340,
  });

  assert.equal(decision.side, "none");
  assert.equal(decision.stake, 0);
  assert.equal(decision.reason, "Balanced caps long-shot underdogs above +300");
});

test("NHL risk-adjusted still allows the same underdog profile", () => {
  const decision = computeBetDecision({
    league: "NHL",
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

test("NBA strategy defaults are tighter than the shared fallback", () => {
  const nba = getBetStrategyConfig("riskAdjusted", { league: "NBA" });
  const nhl = getBetStrategyConfig("riskAdjusted", { league: "NHL" });

  assert.ok(nba.minEdge > nhl.minEdge);
  assert.ok(nba.minExpectedValue > nhl.minExpectedValue);
  assert.equal(nba.maxUnderdogMoneyline, 300);
});

test("league-aware default strategy shifts NBA to conservative while leaving NHL unchanged", () => {
  assert.equal(getDefaultBetStrategyForLeague("NBA"), "capitalPreservation");
  assert.equal(getDefaultBetStrategyForLeague("NHL"), "riskAdjusted");
  assert.equal(getDefaultBetStrategyForLeague("NCAAM"), "riskAdjusted");
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
  assert.ok((trace.baseStakeShareOfBankroll ?? 0) > 0);
  assert.ok((trace.scaledStakeShareOfBankroll ?? 0) >= (trace.cappedStakeShareOfBankroll ?? 0));
  assert.equal(trace.quotedStake, 125);
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
    "capitalPreservation"
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
    "riskAdjusted"
  );

  assert.equal(decisions.length, 2);
  assert.equal(decisions[0].team, "NYK");
  assert.equal(decisions[1].team, "SAC");
});
