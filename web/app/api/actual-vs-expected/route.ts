import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

export async function GET() {
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
        r.home_win
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
        ROW_NUMBER() OVER (
          PARTITION BY game_id
          ORDER BY DATETIME(as_of_utc) DESC, prediction_id DESC
        ) AS rn
      FROM eligible
    )
    SELECT
      game_id,
      game_date_utc,
      home_team,
      away_team,
      as_of_utc,
      prob_home_win,
      predicted_winner,
      home_win,
      CASE
        WHEN (home_win = 1 AND predicted_winner = home_team)
          OR (home_win = 0 AND predicted_winner = away_team)
        THEN 1
        ELSE 0
      END AS model_correct
    FROM ranked
    WHERE rn = 1
    ORDER BY game_date_utc ASC, game_id ASC
    `
  );

  const latest = runSqlJson("SELECT MAX(as_of_utc) AS as_of_utc FROM upcoming_game_forecasts");
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
        ORDER BY u.game_date_utc ASC, u.game_id ASC
        `
      )
    : [];

  return NextResponse.json({ as_of_utc: asOf, historical_rows, upcoming_rows });
}
