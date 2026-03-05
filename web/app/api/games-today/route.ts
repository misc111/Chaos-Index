import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";

type RawTodayGameRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  start_time_utc?: string | null;
};

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function normalizeUtcTimestamp(value: string): string {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$/.test(value)) {
    return value.replace("Z", ":00Z");
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) {
    return `${value}:00Z`;
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(value)) {
    return `${value}Z`;
  }
  return value;
}

function centralDateKeyFromTimestamp(value?: string | null): string | null {
  if (!value) return null;
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return null;

  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(parsed);

  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) return null;
  return `${year}-${month}-${day}`;
}

function centralTodayDateKey(): string {
  const now = new Date();
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);

  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) {
    return now.toISOString().slice(0, 10);
  }
  return `${year}-${month}-${day}`;
}

function dateKeyForRow(row: Pick<RawTodayGameRow, "start_time_utc" | "game_date_utc">): string | null {
  const byStartTime = centralDateKeyFromTimestamp(row.start_time_utc);
  if (byStartTime) return byStartTime;

  const fallback = String(row.game_date_utc || "").trim();
  if (!fallback) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(fallback)) return fallback;

  const byGameDateTimestamp = centralDateKeyFromTimestamp(fallback);
  if (byGameDateTimestamp) return byGameDateTimestamp;

  return fallback.length >= 10 ? fallback.slice(0, 10) : null;
}

function normalizeProbability(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.5;
  return Math.max(0, Math.min(1, numeric));
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  const asOf = latest?.[0]?.as_of_utc;

  if (typeof asOf !== "string" || !asOf.trim()) {
    return NextResponse.json({ league, as_of_utc: null, date_central: centralTodayDateKey(), rows: [] });
  }

  const escapedAsOf = escapeSqlString(asOf);
  const rawRows = runSqlJson(
    `
    SELECT
      u.game_id,
      u.game_date_utc,
      u.home_team,
      u.away_team,
      u.ensemble_prob_home_win AS home_win_probability,
      g.start_time_utc
    FROM upcoming_game_forecasts u
    LEFT JOIN games g ON g.game_id = u.game_id
    WHERE u.as_of_utc = '${escapedAsOf}'
      AND COALESCE(g.status_final, 0) = 0
    ORDER BY
      CASE WHEN g.start_time_utc IS NULL THEN 1 ELSE 0 END,
      DATETIME(g.start_time_utc) ASC,
      u.game_date_utc ASC,
      u.game_id ASC
    `,
    { league }
  ) as RawTodayGameRow[];

  const todayKey = centralTodayDateKey();
  const rows = rawRows
    .map((row) => ({
      ...row,
      home_win_probability: normalizeProbability(row.home_win_probability),
    }))
    .filter((row) => dateKeyForRow(row) === todayKey);

  return NextResponse.json({ league, as_of_utc: asOf, date_central: todayKey, rows });
}
