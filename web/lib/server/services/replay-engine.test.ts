import assert from "node:assert/strict";
import test from "node:test";

import {
  buildReplayDecisionDetail,
  createEmptyReplayStrategySummary,
  defaultReplayDecisionDetail,
  finalizeReplayStrategySummary,
  trackReplayStrategyOutcome,
} from "./replay-engine";

test("replay-engine summary helpers track and finalize strategy outcomes", () => {
  const summary = createEmptyReplayStrategySummary(12);
  const winningDetail = buildReplayDecisionDetail(
    {
      bet: "$100",
      reason: "edge",
      side: "home",
      team: "NYK",
      stake: 100,
      odds: -120,
      modelProbability: 0.62,
      marketProbability: 0.55,
      edge: 0.07,
      expectedValue: 0.04,
    },
    1
  );

  trackReplayStrategyOutcome(summary, winningDetail, "2026-03-01");
  trackReplayStrategyOutcome(summary, defaultReplayDecisionDetail(), "2026-03-02");

  const finalized = finalizeReplayStrategySummary(summary);
  assert.equal(finalized.total_games, 12);
  assert.equal(finalized.suggested_bets, 1);
  assert.equal(finalized.wins, 1);
  assert.equal(finalized.losses, 0);
  assert.equal(finalized.total_risked, 100);
  assert.ok(finalized.total_profit > 0);
  assert.ok(finalized.roi > 0);
  assert.equal(finalized.first_bet_date_central, "2026-03-01");
  assert.equal(finalized.last_bet_date_central, "2026-03-01");
});

test("replay-engine default detail is a no-bet placeholder", () => {
  const detail = defaultReplayDecisionDetail();
  assert.equal(detail.outcome, "no_bet");
  assert.equal(detail.stake, 0);
  assert.equal(detail.profit, 0);
  assert.equal(detail.payout, 0);
});
