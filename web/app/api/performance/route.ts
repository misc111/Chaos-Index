import { NextResponse } from "next/server.js";
import { runSqlJson } from "@/lib/db";
import { leagueFromRequest } from "@/lib/league";
import { canonicalizePredictionModel } from "@/lib/predictions-report";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const scores = runSqlJson(
    `SELECT model_name, game_date_utc, AVG(log_loss) AS log_loss
     FROM model_scores
     GROUP BY model_name, game_date_utc
     ORDER BY game_date_utc ASC`,
    { league }
  ).map((row) => ({
    ...row,
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
  }));

  const change_points = runSqlJson(
    `SELECT model_name, metric_name, method, statistic, threshold, details_json, as_of_utc
     FROM change_points
     ORDER BY as_of_utc DESC
     LIMIT 120`,
    { league }
  ).map((row) => ({
    ...row,
    model_name: canonicalizePredictionModel(String(row.model_name || "")),
  }));

  return NextResponse.json({ league, scores, change_points });
}
