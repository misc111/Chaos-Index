import assert from "node:assert/strict";
import test from "node:test";

import { resolveBetStrategyConfigs, type OptimizableHistoricalBetRow } from "./betting-optimizer";

const DAY_PATTERNS = [
  { favoriteProb: 0.64, favoriteMoneyline: -105, favoriteWin: 1, dogProb: 0.41, dogMoneyline: 150, dogWin: 1 },
  { favoriteProb: 0.61, favoriteMoneyline: -108, favoriteWin: 0, dogProb: 0.37, dogMoneyline: 165, dogWin: 0 },
  { favoriteProb: 0.66, favoriteMoneyline: -110, favoriteWin: 1, dogProb: 0.39, dogMoneyline: 145, dogWin: 0 },
  { favoriteProb: 0.63, favoriteMoneyline: -102, favoriteWin: 1, dogProb: 0.4, dogMoneyline: 152, dogWin: 1 },
  { favoriteProb: 0.67, favoriteMoneyline: -112, favoriteWin: 1, dogProb: 0.38, dogMoneyline: 160, dogWin: 0 },
];

const SAMPLE_ROWS: OptimizableHistoricalBetRow[] = Array.from({ length: 30 }, (_, index) => {
  const pattern = DAY_PATTERNS[index % DAY_PATTERNS.length];
  const day = String(index + 1).padStart(2, "0");

  return [
    {
      game_id: index * 2 + 1,
      date_central: `2026-01-${day}`,
      home_team: `F${index}H`,
      away_team: `F${index}A`,
      home_win_probability: pattern.favoriteProb,
      home_moneyline: pattern.favoriteMoneyline,
      away_moneyline: -115,
      home_win: pattern.favoriteWin,
    },
    {
      game_id: index * 2 + 2,
      date_central: `2026-01-${day}`,
      home_team: `D${index}H`,
      away_team: `D${index}A`,
      home_win_probability: pattern.dogProb,
      home_moneyline: pattern.dogMoneyline,
      away_moneyline: -130,
      home_win: pattern.dogWin,
    },
  ];
}).flat();

const THIN_SAMPLE_ROWS = SAMPLE_ROWS.filter((row) => row.date_central <= "2026-01-10");

test("resolveBetStrategyConfigs derives objective-based profiles from replay history", () => {
  const { strategyConfigs, optimizationSummary } = resolveBetStrategyConfigs(SAMPLE_ROWS);

  assert.equal(strategyConfigs.aggressive.label, "Aggressive");
  assert.equal(strategyConfigs.riskAdjusted.label, "Balanced");
  assert.equal(strategyConfigs.riskAdjusted.optimization_source, "historical_frontier");
  assert.equal(strategyConfigs.aggressive.optimization_source, "historical_frontier");
  assert.equal(strategyConfigs.capitalPreservation.allowUnderdogs, false);
  assert.ok((optimizationSummary.frontier_point_count ?? 0) > 0);
  assert.ok((optimizationSummary.candidate_count ?? 0) > 0);

  assert.ok(strategyConfigs.riskAdjusted.metrics);
  assert.ok(strategyConfigs.aggressive.metrics);
  assert.ok(strategyConfigs.capitalPreservation.metrics);

  assert.ok(
    strategyConfigs.aggressive.metrics!.mean_daily_profit_dollars >=
      strategyConfigs.riskAdjusted.metrics!.mean_daily_profit_dollars
  );
  assert.ok(
    strategyConfigs.riskAdjusted.metrics!.expected_log_growth_per_bet >=
      strategyConfigs.capitalPreservation.metrics!.expected_log_growth_per_bet
  );
  assert.equal(strategyConfigs.aggressive.maxDailyBankrollPercent, null);
  assert.equal(strategyConfigs.aggressive.description.includes("no nightly budget cap"), true);
  assert.ok(strategyConfigs.riskAdjusted.maxDailyBankrollPercent !== null);
  assert.ok(strategyConfigs.capitalPreservation.maxDailyBankrollPercent !== null);
});

test("resolveBetStrategyConfigs falls back to static defaults when replay coverage is thin", () => {
  const { strategyConfigs, optimizationSummary } = resolveBetStrategyConfigs(THIN_SAMPLE_ROWS);

  assert.equal(strategyConfigs.riskAdjusted.optimization_source, "static_fallback");
  assert.equal(strategyConfigs.aggressive.optimization_source, "static_fallback");
  assert.equal(strategyConfigs.capitalPreservation.optimization_source, "static_fallback");
  assert.equal(strategyConfigs.riskAdjusted.metrics, null);
  assert.equal(strategyConfigs.aggressive.maxDailyBankrollPercent, null);
  assert.equal(optimizationSummary.candidate_count, 0);
  assert.equal(optimizationSummary.frontier_point_count, 0);
  assert.match(optimizationSummary.method, /Static defaults active/);
});
