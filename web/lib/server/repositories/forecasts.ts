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
};

export type RawTodayGameRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  forecast_as_of_utc?: string | null;
  start_time_utc?: string | null;
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
  return runSqlJson(
    `SELECT game_id, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner,
            spread_mean, spread_sd, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, per_model_probs_json
     FROM upcoming_game_forecasts
     WHERE as_of_utc = '${escapeSqlString(asOf)}'
     ORDER BY game_date_utc ASC, game_id ASC`,
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
      g.start_time_utc
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
