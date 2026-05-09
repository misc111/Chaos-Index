import assert from "node:assert/strict";
import test from "node:test";

import {
  computeBetDecisionsForSlate,
  formatBetRecommendation,
  REFERENCE_STAKE_DOLLARS,
  type BetInput,
} from "./betting";
import type { BetStrategyRuleConfig } from "./betting-strategy";

const FLAT_STAKE_TEST_CONFIG: BetStrategyRuleConfig = {
  allowUnderdogs: true,
  maxUnderdogMoneyline: null,
  minEdge: 0.03,
  minExpectedValue: 0.02,
  stakeScale: 0.01,
  maxBetBankrollPercent: 0.05,
  maxDailyBankrollPercent: null,
};

const BETTABLE_ROWS: BetInput[] = [
  {
    home_team: "HOME",
    away_team: "AWAY",
    home_win_probability: 0.7,
    home_moneyline: 140,
    away_moneyline: -160,
  },
  {
    home_team: "HOME2",
    away_team: "AWAY2",
    home_win_probability: 0.28,
    home_moneyline: -130,
    away_moneyline: 160,
  },
];

test("accepted bets use the same flat stake regardless of calculated edge size", () => {
  const decisions = computeBetDecisionsForSlate(BETTABLE_ROWS, "riskAdjusted", FLAT_STAKE_TEST_CONFIG);

  assert.equal(decisions[0].stake, REFERENCE_STAKE_DOLLARS);
  assert.equal(decisions[0].bet, `$${REFERENCE_STAKE_DOLLARS} HOME`);
  assert.equal(decisions[1].stake, REFERENCE_STAKE_DOLLARS);
  assert.equal(decisions[1].bet, `$${REFERENCE_STAKE_DOLLARS} AWAY2`);
});

test("rejected games remain no-bets with zero stake", () => {
  const [decision] = computeBetDecisionsForSlate(
    [
      {
        home_team: "FAIRHOME",
        away_team: "FAIRAWAY",
        home_win_probability: 0.5,
        home_moneyline: -110,
        away_moneyline: -110,
      },
    ],
    "riskAdjusted",
    FLAT_STAKE_TEST_CONFIG
  );

  assert.equal(decision.stake, 0);
  assert.equal(decision.bet, "$0");
  assert.match(decision.reason, /No bet/);
});

test("legacy replay reasons are normalized for display", () => {
  assert.equal(
    formatBetRecommendation({
      team: null,
      stake: 0,
      reason: "Adjusted price fair",
    }).reason,
    "No bet: the sportsbook price is close to the model's estimate."
  );

  assert.equal(
    formatBetRecommendation({
      team: "CIN",
      stake: REFERENCE_STAKE_DOLLARS,
      reason: "Underdog underpriced after uncertainty adjustment",
    }).reason,
    "Bet: the model likes this underdog price."
  );
});
