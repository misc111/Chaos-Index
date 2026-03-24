import assert from "node:assert/strict";
import test from "node:test";

import {
  DIAGNOSTIC_FORECAST_SOURCE,
  STORED_FORECAST_SOURCE,
  effectiveReplayOddsAsOfSql,
  historicalForecastCandidatesUnionSql,
  historicalMoneylineSql,
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

test("effectiveReplayOddsAsOfSql uses commence time for historical bundle imports", () => {
  const sql = effectiveReplayOddsAsOfSql("s", "l");
  assert.ok(sql.includes("historical_bundle"));
  assert.ok(sql.includes("historical_manifest"));
  assert.ok(sql.includes("l.commence_time_utc"));
  assert.ok(sql.includes("COALESCE(l.bookmaker_last_update_utc, s.as_of_utc)"));
});
