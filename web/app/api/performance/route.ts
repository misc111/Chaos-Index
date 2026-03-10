import { NextResponse } from "next/server.js";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";
import { canonicalizePredictionModel } from "@/lib/predictions-report";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const scores = runSqlJson(
    `SELECT
       CASE
         WHEN model_name = 'glm_logit' THEN 'glm_ridge'
         ELSE model_name
       END AS model_name,
       game_date_utc,
       AVG(log_loss) AS log_loss
     FROM model_scores
     GROUP BY
       CASE
         WHEN model_name = 'glm_logit' THEN 'glm_ridge'
         ELSE model_name
       END,
       game_date_utc
     ORDER BY game_date_utc ASC`,
    { league }
  ).map((row) => ({
    ...row,
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
  }));

  const change_points = runSqlJson(
    `WITH latest AS (
       SELECT MAX(as_of_utc) AS as_of_utc
       FROM change_points
     )
     SELECT model_name, metric_name, method, statistic, threshold, details_json, as_of_utc
     FROM change_points
     WHERE as_of_utc = (SELECT as_of_utc FROM latest)
     ORDER BY statistic DESC, model_name ASC
     LIMIT 120`,
    { league }
  ).map((row) => ({
    ...row,
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
  }));

  return NextResponse.json({ league, scores, change_points });
}
