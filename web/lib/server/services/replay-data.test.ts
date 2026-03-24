import assert from "node:assert/strict";
import test from "node:test";

import {
  DIAGNOSTIC_FORECAST_SOURCE,
  STORED_FORECAST_SOURCE,
  effectiveReplayOddsAsOfSql,
  historicalForecastCandidatesUnionSql,
  historicalFinalizedGamesCteSql,
  historicalMoneylineSql,
  historicalReplayOddsSelectionSql,
} from "./replay-data";

test("historicalForecastCandidatesUnionSql preserves source names and escapes model filters", () => {
  const sql = historicalForecastCandidatesUnionSql({ modelName: "glm_o'hare" });
  assert.ok(sql.includes(STORED_FORECAST_SOURCE));
  assert.ok(sql.includes(DIAGNOSTIC_FORECAST_SOURCE));
  assert.ok(sql.includes("AND p.model_name = 'glm_o''hare'"));
});

test("historicalMoneylineSql toggles bookmaker title columns", () => {
  const withoutBooks = historicalMoneylineSql(["snapshot_1"]);
  const withBooks = historicalMoneylineSql(["snapshot_1"], { includeBookmakerTitles: true });

  assert.ok(!withoutBooks.includes("home_moneyline_book"));
  assert.ok(withBooks.includes("home_moneyline_book"));
  assert.ok(withBooks.includes("away_moneyline_book"));
});

test("effectiveReplayOddsAsOfSql reads the authoritative effective-odds column", () => {
  const sql = effectiveReplayOddsAsOfSql("l");
  assert.equal(sql, "l.effective_odds_as_of_utc");
});

test("historicalFinalizedGamesCteSql returns a reusable finalized-games CTE", () => {
  const sql = historicalFinalizedGamesCteSql();
  assert.ok(sql.includes("finalized_games AS"));
  assert.ok(sql.includes("COALESCE(g.start_time_utc, r.final_utc, r.game_date_utc || 'T23:59:59Z')"));
  assert.ok(sql.includes("WHERE r.home_win IS NOT NULL"));
});

test("historicalReplayOddsSelectionSql emits both odds snapshot and odds timestamp subqueries", () => {
  const sql = historicalReplayOddsSelectionSql({ league: "NBA", requireConditionSql: "rf.as_of_utc IS NOT NULL" });
  assert.ok(sql.includes("AS odds_snapshot_id"));
  assert.ok(sql.includes("AS odds_as_of_utc"));
  assert.ok(sql.includes("FROM odds_market_lines_effective_as_of l"));
  assert.ok(sql.includes("l.league = 'NBA'"));
  assert.ok(sql.includes("rf.as_of_utc IS NOT NULL"));
  assert.ok(sql.includes("l.effective_odds_as_of_utc"));
  assert.ok(sql.includes("l.snapshot_as_of_utc"));
});
