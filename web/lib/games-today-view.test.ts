import assert from "node:assert/strict";
import test from "node:test";

import { resolveGamesTodayDateView } from "./games-today-view";
import type { GamesTodayRow } from "./types";

function row(gameId: number, startTimeUtc: string): GamesTodayRow {
  return {
    game_id: gameId,
    home_team: "HOME",
    away_team: "AWAY",
    home_win_probability: 0.5,
    start_time_utc: startTimeUtc,
  };
}

test("uses scheduled snapshot rows for the active current-day slate", () => {
  const view = resolveGamesTodayDateView({
    activeDateKey: "2026-03-10",
    todayKey: "2026-03-10",
    upcomingRows: [row(1, "2026-03-10T23:00:00Z"), row(2, "2026-03-11T00:30:00Z")],
    historicalRows: [row(3, "2026-03-10T01:00:00Z")],
  });

  assert.equal(view.mode, "scheduled");
  assert.equal(view.rows.length, 2);
  assert.deepEqual(
    view.rows.map((entry) => entry.game_id),
    [1, 2]
  );
});

test("prefers replay rows for past dates when they exist", () => {
  const view = resolveGamesTodayDateView({
    activeDateKey: "2026-03-09",
    todayKey: "2026-03-10",
    upcomingRows: [row(1, "2026-03-10T00:30:00Z")],
    historicalRows: [row(2, "2026-03-10T00:30:00Z")],
  });

  assert.equal(view.mode, "historical");
  assert.equal(view.rows.length, 1);
  assert.equal(view.rows[0]?.game_id, 2);
});

test("falls back to latest snapshot rows for past dates that are not replayable yet", () => {
  const view = resolveGamesTodayDateView({
    activeDateKey: "2026-03-09",
    todayKey: "2026-03-10",
    upcomingRows: [row(1, "2026-03-09T23:00:00Z"), row(2, "2026-03-10T02:00:00Z"), row(3, "2026-03-10T23:00:00Z")],
    historicalRows: [],
  });

  assert.equal(view.mode, "snapshotFallback");
  assert.equal(view.rows.length, 2);
  assert.deepEqual(
    view.rows.map((entry) => entry.game_id),
    [1, 2]
  );
});

test("returns an explicit empty past-date mode when neither source has rows", () => {
  const view = resolveGamesTodayDateView({
    activeDateKey: "2026-03-09",
    todayKey: "2026-03-10",
    upcomingRows: [row(1, "2026-03-10T23:00:00Z")],
    historicalRows: [],
  });

  assert.equal(view.mode, "emptyPast");
  assert.equal(view.rows.length, 0);
});
