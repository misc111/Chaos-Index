import { runSqlJson } from "@/lib/db";
import { type LeagueCode } from "@/lib/league";
import { escapeSqlString } from "@/lib/server/repositories/sql";

export type RawSnapshotRow = {
  odds_snapshot_id?: string | null;
  as_of_utc?: string | null;
};

export type RawMoneylineRow = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  home_moneyline_book?: string | null;
  away_moneyline_book?: string | null;
};

export type RawPairedMoneylineRow = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  moneyline_book?: string | null;
  home_implied_probability?: number | null;
  away_implied_probability?: number | null;
  overround?: number | null;
};

export type RawOver190Row = {
  odds_snapshot_id?: string | null;
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  over_190_price?: number | null;
  over_190_point?: number | null;
  over_190_book?: string | null;
};

export type RawOddsLine = {
  game_id?: number | null;
  home_team?: string | null;
  away_team?: string | null;
  market_key?: string | null;
  outcome_side?: string | null;
  outcome_point?: number | null;
  outcome_price?: number | null;
  bookmaker_title?: string | null;
};

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

function pairedMoneylineSql(snapshotIds: string[]): string {
  const inList = snapshotIds.map((snapshotId) => `'${escapeSqlString(snapshotId)}'`).join(", ");
  return `
    WITH paired_by_book AS (
      SELECT
        odds_snapshot_id,
        game_id,
        home_team,
        away_team,
        bookmaker_key,
        MAX(bookmaker_title) AS bookmaker_title,
        MAX(bookmaker_last_update_utc) AS bookmaker_last_update_utc,
        MAX(CASE WHEN outcome_side = 'home' THEN outcome_price END) AS home_moneyline,
        MAX(CASE WHEN outcome_side = 'away' THEN outcome_price END) AS away_moneyline,
        MAX(CASE WHEN outcome_side = 'home' THEN implied_probability END) AS home_implied_probability,
        MAX(CASE WHEN outcome_side = 'away' THEN implied_probability END) AS away_implied_probability
      FROM odds_market_lines
      WHERE odds_snapshot_id IN (${inList})
        AND market_key = 'h2h'
        AND outcome_side IN ('home', 'away')
        AND outcome_price IS NOT NULL
      GROUP BY odds_snapshot_id, game_id, home_team, away_team, bookmaker_key
      HAVING home_moneyline IS NOT NULL
        AND away_moneyline IS NOT NULL
    ),
    ranked AS (
      SELECT
        *,
        CASE
          WHEN home_implied_probability IS NOT NULL AND away_implied_probability IS NOT NULL
            THEN home_implied_probability + away_implied_probability
          ELSE NULL
        END AS overround,
        ROW_NUMBER() OVER (
          PARTITION BY
            odds_snapshot_id,
            COALESCE(CAST(game_id AS TEXT), home_team || '|' || away_team)
          ORDER BY
            CASE
              WHEN home_implied_probability IS NOT NULL AND away_implied_probability IS NOT NULL
                THEN home_implied_probability + away_implied_probability
              ELSE 999
            END ASC,
            DATETIME(bookmaker_last_update_utc) DESC,
            bookmaker_title ASC
        ) AS rn
      FROM paired_by_book
    )
    SELECT
      odds_snapshot_id,
      game_id,
      home_team,
      away_team,
      home_moneyline,
      away_moneyline,
      bookmaker_title AS moneyline_book,
      home_implied_probability,
      away_implied_probability,
      overround
    FROM ranked
    WHERE rn = 1
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

export function getOddsSnapshots(league: LeagueCode): RawSnapshotRow[] {
  return runSqlJson(
    `
    SELECT odds_snapshot_id, as_of_utc
    FROM odds_snapshots
    WHERE league = '${escapeSqlString(league)}'
    ORDER BY DATETIME(as_of_utc) DESC
    `,
    { league }
  ) as RawSnapshotRow[];
}

export function getLatestOddsSnapshot(league: LeagueCode): RawSnapshotRow | null {
  const rows = runSqlJson(
    `
    SELECT odds_snapshot_id, as_of_utc
    FROM odds_snapshots
    WHERE league = '${escapeSqlString(league)}'
    ORDER BY DATETIME(as_of_utc) DESC
    LIMIT 1
    `,
    { league }
  ) as RawSnapshotRow[];
  return rows[0] || null;
}

export function getMoneylineRowsForSnapshots(league: LeagueCode, snapshotIds: string[]): RawMoneylineRow[] {
  if (!snapshotIds.length) return [];
  return runSqlJson(moneylineSql(snapshotIds), { league }) as RawMoneylineRow[];
}

export function getPairedMoneylineRowsForSnapshots(league: LeagueCode, snapshotIds: string[]): RawPairedMoneylineRow[] {
  if (!snapshotIds.length) return [];
  return runSqlJson(pairedMoneylineSql(snapshotIds), { league }) as RawPairedMoneylineRow[];
}

export function getOver190RowsForSnapshots(league: LeagueCode, snapshotIds: string[]): RawOver190Row[] {
  if (!snapshotIds.length || league !== "NBA") return [];
  return runSqlJson(over190Sql(snapshotIds), { league }) as RawOver190Row[];
}

export function getMarketLinesForSnapshot(league: LeagueCode, snapshotId: string, gameIds: number[]): RawOddsLine[] {
  const gameIdClause = gameIds.length ? `AND game_id IN (${gameIds.join(",")})` : "AND 1 = 0";
  if (!snapshotId) return [];
  return runSqlJson(
    `
    SELECT
      game_id,
      home_team,
      away_team,
      market_key,
      outcome_side,
      outcome_point,
      outcome_price,
      bookmaker_title
    FROM odds_market_lines
    WHERE odds_snapshot_id = '${escapeSqlString(snapshotId)}'
      AND market_key IN ('h2h', 'spreads', 'totals')
      AND outcome_price IS NOT NULL
      ${gameIdClause}
    ORDER BY game_id ASC, market_key ASC, bookmaker_title ASC
    `,
    { league }
  ) as RawOddsLine[];
}
