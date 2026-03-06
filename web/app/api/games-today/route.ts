import { NextResponse } from "next/server";
import { getHistoricalReplayGames } from "@/lib/bet-history";
import { runSqlJson } from "@/lib/db";
import { centralDateKeyFromTimestamp, centralTodayDateKey, dateKeyForScheduledGame } from "@/lib/games-today";
import { leagueFromRequest } from "@/lib/league";

// Maintainer note: this route reads the target league from ?league=...
// and must stay dynamic in the live dashboard. Making it static causes
// Next.js to freeze one league's payload and serve it to both NHL/NBA.
export const dynamic = "force-dynamic";

type RawTodayGameRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  forecast_as_of_utc?: string | null;
  start_time_utc?: string | null;
};

type RawSnapshotRow = {
  odds_snapshot_id?: string | null;
  as_of_utc?: string | null;
};

type RawMoneylineRow = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  home_moneyline_book?: string | null;
  away_moneyline_book?: string | null;
};

type RawOver190Row = {
  odds_snapshot_id?: string | null;
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

function normalizeProbability(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.5;
  return Math.max(0, Math.min(1, numeric));
}

function moneylineSql(snapshotIds: string[]): string {
  const inList = snapshotIds.map((snapshotId) => `'${escapeSqlString(snapshotId)}'`).join(", ");

  return `
    WITH ranked AS (
      SELECT
        odds_snapshot_id,
        game_id,
        home_team,
        away_team,
        outcome_side,
        outcome_price,
        bookmaker_title,
        ROW_NUMBER() OVER (
          PARTITION BY
            odds_snapshot_id,
            COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team),
            outcome_side
          ORDER BY outcome_price DESC, DATETIME(bookmaker_last_update_utc) DESC, line_id DESC
        ) AS rn
      FROM odds_market_lines
      WHERE odds_snapshot_id IN (${inList})
        AND market_key = 'h2h'
        AND outcome_side IN ('home', 'away')
        AND outcome_price IS NOT NULL
    )
    SELECT
      odds_snapshot_id,
      game_id,
      home_team,
      away_team,
      MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN outcome_price END) AS home_moneyline,
      MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN outcome_price END) AS away_moneyline,
      MAX(CASE WHEN outcome_side = 'home' AND rn = 1 THEN bookmaker_title END) AS home_moneyline_book,
      MAX(CASE WHEN outcome_side = 'away' AND rn = 1 THEN bookmaker_title END) AS away_moneyline_book
    FROM ranked
    GROUP BY odds_snapshot_id, game_id, home_team, away_team
  `;
}

function over190Sql(snapshotIds: string[]): string {
  const inList = snapshotIds.map((snapshotId) => `'${escapeSqlString(snapshotId)}'`).join(", ");

  return `
    WITH ranked AS (
      SELECT
        odds_snapshot_id,
        game_id,
        home_team,
        away_team,
        outcome_price,
        outcome_point,
        bookmaker_title,
        ROW_NUMBER() OVER (
          PARTITION BY
            odds_snapshot_id,
            COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team)
          ORDER BY outcome_point ASC, DATETIME(bookmaker_last_update_utc) DESC, line_id DESC
        ) AS rn
      FROM odds_market_lines
      WHERE odds_snapshot_id IN (${inList})
        AND market_key = 'alternate_totals'
        AND outcome_side = 'over'
        AND outcome_point >= 190.0
        AND outcome_price IS NOT NULL
    )
    SELECT
      odds_snapshot_id,
      game_id,
      home_team,
      away_team,
      MAX(CASE WHEN rn = 1 THEN outcome_price END) AS over_190_price,
      MAX(CASE WHEN rn = 1 THEN outcome_point END) AS over_190_point,
      MAX(CASE WHEN rn = 1 THEN bookmaker_title END) AS over_190_book
    FROM ranked
    GROUP BY odds_snapshot_id, game_id, home_team, away_team
  `;
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const historicalReplay = getHistoricalReplayGames(league);
  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  const asOf = latest?.[0]?.as_of_utc;

  if (typeof asOf !== "string" || !asOf.trim()) {
    return NextResponse.json({
      league,
      as_of_utc: null,
      date_central: centralTodayDateKey(),
      historical_coverage_start_central: historicalReplay.coverage_start_central,
      historical_rows: historicalReplay.rows,
      rows: [],
    });
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
      u.as_of_utc AS forecast_as_of_utc,
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

  const rows = rawRows
    .map((row) => ({
      ...row,
      home_win_probability: normalizeProbability(row.home_win_probability),
    }));

  const snapshotRows = runSqlJson(
    `
    SELECT odds_snapshot_id, as_of_utc
    FROM odds_snapshots
    WHERE league = '${escapeSqlString(league)}'
    ORDER BY DATETIME(as_of_utc) DESC
    `,
    { league }
  ) as RawSnapshotRow[];

  const latestOddsSnapshotByDate = new Map<string, { odds_snapshot_id: string; as_of_utc: string }>();
  for (const row of snapshotRows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const asOfUtc = String(row.as_of_utc || "").trim();
    const snapshotDateKey = centralDateKeyFromTimestamp(asOfUtc);
    if (!snapshotId || !asOfUtc || !snapshotDateKey || latestOddsSnapshotByDate.has(snapshotDateKey)) {
      continue;
    }
    latestOddsSnapshotByDate.set(snapshotDateKey, { odds_snapshot_id: snapshotId, as_of_utc: asOfUtc });
  }

  const neededSnapshotIds = Array.from(
    new Set(
      [
        ...rows.map((row) => {
          const rowDateKey = dateKeyForScheduledGame(row);
          return rowDateKey ? latestOddsSnapshotByDate.get(rowDateKey)?.odds_snapshot_id || "" : "";
        }),
        ...historicalReplay.rows.map((row) => latestOddsSnapshotByDate.get(row.date_central)?.odds_snapshot_id || ""),
      ].filter(Boolean)
    )
  );

  const moneylineRows = neededSnapshotIds.length
    ? (runSqlJson(moneylineSql(neededSnapshotIds), { league }) as RawMoneylineRow[])
    : [];

  // Maintainer note: Games Today uses the latest odds snapshot stored on the
  // selected Central date. That intentionally leaves future dates blank until
  // that day's own refresh exists, instead of carrying today's snapshot forward.
  const over190Rows =
    neededSnapshotIds.length && league === "NBA"
      ? (runSqlJson(over190Sql(neededSnapshotIds), { league }) as RawOver190Row[])
      : [];

  const moneylineByGameId = new Map<string, RawMoneylineRow>();
  const moneylineByTeamKey = new Map<string, RawMoneylineRow>();
  for (const row of moneylineRows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const gameId = Number(row.game_id);
    if (snapshotId && Number.isFinite(gameId)) {
      moneylineByGameId.set(`${snapshotId}::${gameId}`, row);
    }
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (snapshotId && homeTeam && awayTeam) {
      moneylineByTeamKey.set(`${snapshotId}::${homeTeam}|${awayTeam}`, row);
    }
  }

  const over190ByGameId = new Map<string, RawOver190Row>();
  const over190ByTeamKey = new Map<string, RawOver190Row>();
  for (const row of over190Rows) {
    const snapshotId = String(row.odds_snapshot_id || "").trim();
    const gameId = Number(row.game_id);
    if (snapshotId && Number.isFinite(gameId)) {
      over190ByGameId.set(`${snapshotId}::${gameId}`, row);
    }
    const homeTeam = String(row.home_team || "").trim();
    const awayTeam = String(row.away_team || "").trim();
    if (snapshotId && homeTeam && awayTeam) {
      over190ByTeamKey.set(`${snapshotId}::${homeTeam}|${awayTeam}`, row);
    }
  }

  function buildHistoricalRow(row: (typeof historicalReplay.rows)[number]) {
    const daySnapshot = latestOddsSnapshotByDate.get(row.date_central);
    const snapshotId = daySnapshot?.odds_snapshot_id || "";
    const moneylineMatch =
      moneylineByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      moneylineByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);
    const over190Match =
      over190ByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      over190ByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);

    return {
      game_id: row.game_id,
      game_date_utc: row.date_central,
      home_team: row.home_team,
      away_team: row.away_team,
      home_win_probability: row.home_win_probability,
      forecast_as_of_utc: row.forecast_as_of_utc,
      odds_as_of_utc: daySnapshot?.as_of_utc || null,
      start_time_utc: row.start_time_utc,
      home_moneyline: moneylineMatch?.home_moneyline ?? null,
      away_moneyline: moneylineMatch?.away_moneyline ?? null,
      home_moneyline_book: moneylineMatch?.home_moneyline_book ?? null,
      away_moneyline_book: moneylineMatch?.away_moneyline_book ?? null,
      over_190_price: over190Match?.over_190_price ?? null,
      over_190_point: over190Match?.over_190_point ?? null,
      over_190_book: over190Match?.over_190_book ?? null,
      replay_decision: row.replay_decision ?? null,
    };
  }

  const enrichedRows = rows.map((row) => {
    const rowDateKey = dateKeyForScheduledGame(row);
    const daySnapshot = rowDateKey ? latestOddsSnapshotByDate.get(rowDateKey) : null;
    const snapshotId = daySnapshot?.odds_snapshot_id || "";
    const moneylineMatch =
      moneylineByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      moneylineByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);
    const over190Match =
      over190ByGameId.get(`${snapshotId}::${Number(row.game_id)}`) ||
      over190ByTeamKey.get(`${snapshotId}::${row.home_team}|${row.away_team}`);
    return {
      ...row,
      odds_as_of_utc: daySnapshot?.as_of_utc || null,
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
    date_central: centralTodayDateKey(),
    // Maintainer note: the client intentionally gets two row pools:
    // `rows` for the latest upcoming snapshot, `historical_rows` for past-date
    // replay navigation. That keeps one UI while supporting both NHL/NBA.
    historical_coverage_start_central: historicalReplay.coverage_start_central,
    historical_rows: historicalReplay.rows.map(buildHistoricalRow),
    odds_as_of_utc: latestOddsSnapshotByDate.get(centralTodayDateKey())?.as_of_utc || null,
    rows: enrichedRows,
  });
}
