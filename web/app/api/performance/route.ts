import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";

export async function GET() {
  const scores = runSqlJson(
    `SELECT model_name, game_date_utc, AVG(log_loss) AS log_loss
     FROM model_scores
     GROUP BY model_name, game_date_utc
     ORDER BY game_date_utc ASC`
  );

  const change_points = runSqlJson(
    `SELECT model_name, metric_name, method, statistic, threshold, details_json, as_of_utc
     FROM change_points
     ORDER BY as_of_utc DESC
     LIMIT 120`
  );

  return NextResponse.json({ scores, change_points });
}
