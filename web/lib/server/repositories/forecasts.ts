import { runSqlJson } from "@/lib/db";
import { type LeagueCode } from "@/lib/league";
import { escapeSqlString } from "@/lib/server/repositories/sql";

export type RawPredictionRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  ensemble_prob_home_win: number;
  predicted_winner?: string | null;
  spread_mean?: number | null;
  spread_sd?: number | null;
  bayes_ci_low?: number | null;
  bayes_ci_high?: number | null;
  uncertainty_flags_json?: string | null;
  per_model_probs_json?: string | null;
  odds_snapshot_id?: string | null;
  odds_as_of_utc?: string | null;
};

export type RawTodayGameRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  forecast_as_of_utc?: string | null;
  start_time_utc?: string | null;
  per_model_probs_json?: string | null;
};

export type RawGamesTodaySnapshotRow = RawTodayGameRow & {
  status_final?: number | null;
  odds_snapshot_id?: string | null;
  odds_as_of_utc?: string | null;
};

export type RawTeamNameRow = {
  team_abbrev?: string | null;
  team_name?: string | null;
};

export function getLatestUpcomingAsOf(league: LeagueCode): string | null {
  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  return typeof latest?.[0]?.as_of_utc === "string" ? latest[0].as_of_utc : null;
}

export function getPredictionRows(league: LeagueCode, asOf: string): RawPredictionRow[] {
  const escapedLeague = escapeSqlString(league);
  const escapedAsOf = escapeSqlString(asOf);
  return runSqlJson(
    `
    SELECT
      u.game_id,
      u.game_date_utc,
      u.home_team,
      u.away_team,
      u.ensemble_prob_home_win,
      u.predicted_winner,
      u.spread_mean,
      u.spread_sd,
      u.bayes_ci_low,
      u.bayes_ci_high,
      u.uncertainty_flags_json,
      u.per_model_probs_json,
      (
        SELECT s.odds_snapshot_id
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(COALESCE(g.start_time_utc, u.game_date_utc || 'T23:59:59Z'))
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = u.game_id)
                OR (l.home_team = u.home_team AND l.away_team = u.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_snapshot_id,
      (
        SELECT s.as_of_utc
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(COALESCE(g.start_time_utc, u.game_date_utc || 'T23:59:59Z'))
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = u.game_id)
                OR (l.home_team = u.home_team AND l.away_team = u.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_as_of_utc
    FROM upcoming_game_forecasts u
    LEFT JOIN games g ON g.game_id = u.game_id
    WHERE u.as_of_utc = '${escapedAsOf}'
    ORDER BY u.game_date_utc ASC, u.game_id ASC
    `,
    { league }
  ) as RawPredictionRow[];
}

export function getScheduledTodayRows(league: LeagueCode, asOf: string): RawTodayGameRow[] {
  return runSqlJson(
    `
    SELECT
      u.game_id,
      u.game_date_utc,
      u.home_team,
      u.away_team,
      u.ensemble_prob_home_win AS home_win_probability,
      u.as_of_utc AS forecast_as_of_utc,
      g.start_time_utc,
      u.per_model_probs_json
    FROM upcoming_game_forecasts u
    LEFT JOIN games g ON g.game_id = u.game_id
    WHERE u.as_of_utc = '${escapeSqlString(asOf)}'
      AND COALESCE(g.status_final, 0) = 0
    ORDER BY
      CASE WHEN g.start_time_utc IS NULL THEN 1 ELSE 0 END,
      DATETIME(g.start_time_utc) ASC,
      u.game_date_utc ASC,
      u.game_id ASC
    `,
    { league }
  ) as RawTodayGameRow[];
}

export function getGamesTodaySnapshotRows(league: LeagueCode, asOf: string): RawGamesTodaySnapshotRow[] {
  const escapedLeague = escapeSqlString(league);
  const escapedAsOf = escapeSqlString(asOf);

  return runSqlJson(
    `
    SELECT
      u.game_id,
      u.game_date_utc,
      u.home_team,
      u.away_team,
      u.ensemble_prob_home_win AS home_win_probability,
      u.as_of_utc AS forecast_as_of_utc,
      g.start_time_utc,
      u.per_model_probs_json,
      COALESCE(g.status_final, 0) AS status_final,
      (
        SELECT s.odds_snapshot_id
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(COALESCE(g.start_time_utc, u.game_date_utc || 'T23:59:59Z'))
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = u.game_id)
                OR (l.home_team = u.home_team AND l.away_team = u.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_snapshot_id,
      (
        SELECT s.as_of_utc
        FROM odds_snapshots s
        WHERE s.league = '${escapedLeague}'
          AND DATETIME(s.as_of_utc) <= DATETIME(COALESCE(g.start_time_utc, u.game_date_utc || 'T23:59:59Z'))
          AND EXISTS (
            SELECT 1
            FROM odds_market_lines l
            WHERE l.odds_snapshot_id = s.odds_snapshot_id
              AND l.market_key = 'h2h'
              AND (
                (l.game_id IS NOT NULL AND l.game_id = u.game_id)
                OR (l.home_team = u.home_team AND l.away_team = u.away_team)
              )
          )
        ORDER BY DATETIME(s.as_of_utc) DESC
        LIMIT 1
      ) AS odds_as_of_utc
    FROM upcoming_game_forecasts u
    LEFT JOIN games g ON g.game_id = u.game_id
    WHERE u.as_of_utc = '${escapedAsOf}'
    ORDER BY
      CASE WHEN g.start_time_utc IS NULL THEN 1 ELSE 0 END,
      DATETIME(g.start_time_utc) ASC,
      u.game_date_utc ASC,
      u.game_id ASC
    `,
    { league }
  ) as RawGamesTodaySnapshotRow[];
}

export function getLatestTeamNames(league: LeagueCode): RawTeamNameRow[] {
  return runSqlJson(
    `
    SELECT team_abbrev, team_name
    FROM teams
    WHERE league = '${escapeSqlString(league)}'
      AND as_of_utc = (
        SELECT MAX(as_of_utc)
        FROM teams
        WHERE league = '${escapeSqlString(league)}'
      )
    `,
    { league }
  ) as RawTeamNameRow[];
}
