import { dateKeyForScheduledGame } from "@/lib/games-today";
import type { GamesTodayRow } from "@/lib/types";

export type GamesTodayViewMode = "scheduled" | "historical" | "snapshotFallback" | "emptyPast";

export type GamesTodayDateView = {
  isPastDate: boolean;
  mode: GamesTodayViewMode;
  rows: GamesTodayRow[];
};

function rowsForDate(rows: GamesTodayRow[], dateKey: string): GamesTodayRow[] {
  return rows.filter((row) => dateKeyForScheduledGame(row) === dateKey);
}

export function resolveGamesTodayDateView(params: {
  activeDateKey: string;
  todayKey: string;
  upcomingRows: GamesTodayRow[];
  historicalRows: GamesTodayRow[];
}): GamesTodayDateView {
  const { activeDateKey, todayKey, upcomingRows, historicalRows } = params;
  const isPastDate = activeDateKey < todayKey;
  const snapshotRows = rowsForDate(upcomingRows, activeDateKey);

  if (!isPastDate) {
    return {
      isPastDate,
      mode: "scheduled",
      rows: snapshotRows,
    };
  }

  const replayRows = rowsForDate(historicalRows, activeDateKey);
  if (replayRows.length > 0) {
    return {
      isPastDate,
      mode: "historical",
      rows: replayRows,
    };
  }

  if (snapshotRows.length > 0) {
    return {
      isPastDate,
      mode: "snapshotFallback",
      rows: snapshotRows,
    };
  }

  return {
    isPastDate,
    mode: "emptyPast",
    rows: [],
  };
}
