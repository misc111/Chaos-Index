import assert from "node:assert/strict";
import test from "node:test";

import { resolveBetStrategyConfigs, type OptimizableHistoricalBetRow } from "./betting-optimizer";

const SAMPLE_ROWS: OptimizableHistoricalBetRow[] = [
  {
    game_id: 1,
    date_central: "2026-01-01",
    home_team: "A",
    away_team: "B",
    home_win_probability: 0.64,
    home_moneyline: -105,
    away_moneyline: -115,
    home_win: 1,
  },
  {
    game_id: 2,
    date_central: "2026-01-01",
    home_team: "C",
    away_team: "D",
    home_win_probability: 0.41,
    home_moneyline: 150,
    away_moneyline: -130,
    home_win: 1,
  },
  {
    game_id: 3,
    date_central: "2026-01-02",
    home_team: "E",
    away_team: "F",
    home_win_probability: 0.61,
    home_moneyline: -108,
    away_moneyline: -112,
    home_win: 0,
  },
  {
    game_id: 4,
    date_central: "2026-01-02",
    home_team: "G",
    away_team: "H",
    home_win_probability: 0.37,
    home_moneyline: 165,
    away_moneyline: -145,
    home_win: 0,
  },
  {
    game_id: 5,
    date_central: "2026-01-03",
    home_team: "I",
    away_team: "J",
    home_win_probability: 0.66,
    home_moneyline: -110,
    away_moneyline: 100,
    home_win: 1,
  },
  {
    game_id: 6,
    date_central: "2026-01-03",
    home_team: "K",
    away_team: "L",
    home_win_probability: 0.39,
    home_moneyline: 145,
    away_moneyline: -125,
    home_win: 0,
  },
  {
    game_id: 7,
    date_central: "2026-01-04",
    home_team: "M",
    away_team: "N",
    home_win_probability: 0.63,
    home_moneyline: -102,
    away_moneyline: -118,
    home_win: 1,
  },
  {
    game_id: 8,
    date_central: "2026-01-04",
    home_team: "O",
    away_team: "P",
    home_win_probability: 0.4,
    home_moneyline: 152,
    away_moneyline: -132,
    home_win: 1,
  },
  {
    game_id: 9,
    date_central: "2026-01-05",
    home_team: "Q",
    away_team: "R",
    home_win_probability: 0.67,
    home_moneyline: -112,
    away_moneyline: 102,
    home_win: 1,
  },
  {
    game_id: 10,
    date_central: "2026-01-05",
    home_team: "S",
    away_team: "T",
    home_win_probability: 0.38,
    home_moneyline: 160,
    away_moneyline: -140,
    home_win: 0,
  },
  {
    game_id: 11,
    date_central: "2026-01-06",
    home_team: "U",
    away_team: "V",
    home_win_probability: 0.62,
    home_moneyline: -104,
    away_moneyline: -116,
    home_win: 1,
  },
  {
    game_id: 12,
    date_central: "2026-01-06",
    home_team: "W",
    away_team: "X",
    home_win_probability: 0.36,
    home_moneyline: 170,
    away_moneyline: -150,
    home_win: 0,
  },
];

test("resolveBetStrategyConfigs derives objective-based profiles from replay history", () => {
  const { strategyConfigs, optimizationSummary } = resolveBetStrategyConfigs(SAMPLE_ROWS);

  assert.equal(strategyConfigs.aggressive.label, "Aggressive");
  assert.equal(strategyConfigs.riskAdjusted.optimization_source, "historical_frontier");
  assert.equal(strategyConfigs.aggressive.optimization_source, "historical_frontier");
  assert.equal(strategyConfigs.capitalPreservation.allowUnderdogs, false);
  assert.ok((optimizationSummary.frontier_point_count ?? 0) > 0);
  assert.ok((optimizationSummary.candidate_count ?? 0) > 0);

  assert.ok(strategyConfigs.riskAdjusted.metrics);
  assert.ok(strategyConfigs.aggressive.metrics);
  assert.ok(strategyConfigs.capitalPreservation.metrics);

  assert.ok(
    strategyConfigs.aggressive.metrics!.mean_daily_profit_units >=
      strategyConfigs.riskAdjusted.metrics!.mean_daily_profit_units
  );
  assert.ok(
    strategyConfigs.riskAdjusted.metrics!.sharpe_ratio >= strategyConfigs.aggressive.metrics!.sharpe_ratio
  );
});
