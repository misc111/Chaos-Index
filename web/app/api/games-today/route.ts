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

type RawMoneylineRow = {
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  home_moneyline_book?: string | null;
  away_moneyline_book?: string | null;
};

type RawOver190Row = {
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  over_190_price?: number | null;
  over_190_point?: number | null;
  over_190_book?: string | null;
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

  const latestOddsSnapshot = runSqlJson(
    `
    SELECT odds_snapshot_id, as_of_utc
    FROM odds_snapshots
    WHERE league = '${escapeSqlString(league)}'
    ORDER BY DATETIME(as_of_utc) DESC
    LIMIT 1
    `,
    { league }
  );

  const oddsSnapshotId = typeof latestOddsSnapshot?.[0]?.odds_snapshot_id === "string" ? latestOddsSnapshot[0].odds_snapshot_id : "";
  const oddsAsOfUtc = typeof latestOddsSnapshot?.[0]?.as_of_utc === "string" ? latestOddsSnapshot[0].as_of_utc : null;
  const escapedSnapshotId = oddsSnapshotId ? escapeSqlString(oddsSnapshotId) : "";

  const moneylineRows = escapedSnapshotId
    ? (runSqlJson(
        `
        WITH ranked AS (
          SELECT
            game_id,
            home_team,
            away_team,
            outcome_side,
            outcome_price,
            bookmaker_title,
            ROW_NUMBER() OVER (
              PARTITION BY
                COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team),
                outcome_side
              ORDER BY outcome_price DESC, DATETIME(bookmaker_last_update_utc) DESC, line_id DESC
            ) AS rn
          FROM odds_market_lines
          WHERE odds_snapshot_id = '${escapedSnapshotId}'
            AND market_key = 'h2h'
            AND outcome_side IN ('home', 'away')
            AND outcome_price IS NOT NULL
        )
        SELECT
          game_id,
          home_team,
          away_team,
          MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN outcome_price END) AS home_moneyline,
          MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN outcome_price END) AS away_moneyline,
          MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN bookmaker_title END) AS home_moneyline_book,
          MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN bookmaker_title END) AS away_moneyline_book
        FROM ranked
        GROUP BY game_id, home_team, away_team
        `,
        { league }
      ) as RawMoneylineRow[])
    : [];

  const over190Rows =
    escapedSnapshotId && league === "NBA"
      ? (runSqlJson(
          `
          WITH ranked AS (
            SELECT
              game_id,
              home_team,
              away_team,
              outcome_price,
              bookmaker_title,
              ROW_NUMBER() OVER (
                PARTITION BY COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team)
                ORDER BY outcome_point ASC, DATETIME(bookmaker_last_update_utc) DESC, line_id DESC
              ) AS rn
            FROM odds_market_lines
            WHERE odds_snapshot_id = '${escapedSnapshotId}'
              AND market_key = 'alternate_totals'
              AND outcome_side = 'over'
              AND outcome_point >= 190.0
              AND outcome_price IS NOT NULL
            )
          SELECT
            game_id,
            home_team,
            away_team,
            MAX(CASE WHEN rn = 1 THEN outcome_price END) AS over_190_price,
            MAX(CASE WHEN rn = 1 THEN outcome_point END) AS over_190_point,
            MAX(CASE WHEN rn = 1 THEN bookmaker_title END) AS over_190_book
          FROM ranked
          GROUP BY game_id, home_team, away_team
          `,
          { league }
        ) as RawOver190Row[])
      : [];

  const moneylineByGameId = new Map<number, RawMoneylineRow>();
  const moneylineByTeamKey = new Map<string, RawMoneylineRow>();
  for (const row of moneylineRows) {
    const gameId = Number(row.game_id);
    if (Number.isFinite(gameId)) {
      moneylineByGameId.set(gameId, row);
    }
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (homeTeam && awayTeam) {
      moneylineByTeamKey.set(`${homeTeam}|${awayTeam}`, row);
    }
  }

  const over190ByGameId = new Map<number, RawOver190Row>();
  const over190ByTeamKey = new Map<string, RawOver190Row>();
  for (const row of over190Rows) {
    const gameId = Number(row.game_id);
    if (Number.isFinite(gameId)) {
      over190ByGameId.set(gameId, row);
    }
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (homeTeam && awayTeam) {
      over190ByTeamKey.set(`${homeTeam}|${awayTeam}`, row);
    }
  }

  const enrichedRows = rows.map((row) => {
    const moneylineMatch =
      moneylineByGameId.get(Number(row.game_id)) ||
      moneylineByTeamKey.get(`${row.home_team}|${row.away_team}`);
    const over190Match =
      over190ByGameId.get(Number(row.game_id)) ||
      over190ByTeamKey.get(`${row.home_team}|${row.away_team}`);
    return {
      ...row,
      home_moneyline: moneylineMatch?.home_moneyline ?? null,
      away_moneyline: moneylineMatch?.away_moneyline ?? null,
      home_moneyline_book: moneylineMatch?.home_moneyline_book ?? null,
      away_moneyline_book: moneylineMatch?.away_moneyline_book ?? null,
      over_190_price: over190Match?.over_190_price ?? null,
      over_190_point: over190Match?.over_190_point ?? null,
      over_190_book: over190Match?.over_190_book ?? null,
    };
  });

  return NextResponse.json({
    league,
    as_of_utc: asOf,
    date_central: todayKey,
    odds_as_of_utc: oddsAsOfUtc,
    rows: enrichedRows,
  });
}
