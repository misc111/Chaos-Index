import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";

export const dynamic = "force-static";

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const historical_rows = runSqlJson(
    `
    WITH eligible AS (
      SELECT
        p.prediction_id,
        p.game_id,
        p.as_of_utc,
        p.game_date_utc,
        p.home_team,
        p.away_team,
        p.pred_winner,
        p.prob_home_win,
        r.home_win,
        r.final_utc
      FROM predictions p
      JOIN results r ON r.game_id = p.game_id
      WHERE p.model_name = 'ensemble'
        AND DATETIME(p.as_of_utc) <= COALESCE(
          DATETIME(r.final_utc),
          DATETIME(r.game_date_utc || 'T23:59:59')
        )
    ),
    ranked AS (
      SELECT
        prediction_id,
        game_id,
        as_of_utc,
        game_date_utc,
        home_team,
        away_team,
        prob_home_win,
        COALESCE(pred_winner, CASE WHEN prob_home_win >= 0.5 THEN home_team ELSE away_team END) AS predicted_winner,
        home_win,
        final_utc,
        ROW_NUMBER() OVER (
          PARTITION BY game_id
          ORDER BY DATETIME(as_of_utc) DESC, prediction_id DESC
        ) AS rn
      FROM eligible
    )
    SELECT
      r.game_id,
      r.game_date_utc,
      r.home_team,
      r.away_team,
      r.as_of_utc,
      r.prob_home_win,
      r.predicted_winner,
      r.home_win,
      r.final_utc,
      g.start_time_utc,
      CASE
        WHEN r.prob_home_win >= 0.45 AND r.prob_home_win <= 0.55 THEN 1
        ELSE 0
      END AS is_toss_up,
      CASE
        WHEN r.prob_home_win >= 0.45 AND r.prob_home_win <= 0.55 THEN NULL
        WHEN (r.home_win = 1 AND r.predicted_winner = r.home_team)
          OR (r.home_win = 0 AND r.predicted_winner = r.away_team)
        THEN 1
        ELSE 0
      END AS model_correct
    FROM ranked r
    LEFT JOIN games g ON g.game_id = r.game_id
    WHERE r.rn = 1
    ORDER BY r.game_date_utc ASC, r.game_id ASC
    `,
    { league }
  );

  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts", { league });
  const asOf = latest?.[0]?.as_of_utc;
  const escapedAsOf = typeof asOf === "string" ? escapeSqlString(asOf) : null;

  const upcoming_rows = escapedAsOf
    ? runSqlJson(
        `
        SELECT
          u.game_id,
          u.game_date_utc,
          u.home_team,
          u.away_team,
          u.as_of_utc,
          u.ensemble_prob_home_win,
          u.predicted_winner,
          g.start_time_utc
        FROM upcoming_game_forecasts u
        LEFT JOIN games g ON g.game_id = u.game_id
        WHERE u.as_of_utc = '${escapedAsOf}'
          AND u.game_date_utc >= DATE('now')
          AND COALESCE(g.status_final, 0) = 0
        ORDER BY u.game_date_utc ASC, u.game_id ASC
        `,
        { league }
      )
    : [];

  return NextResponse.json({ league, as_of_utc: asOf, historical_rows, upcoming_rows });
}
