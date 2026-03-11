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

  const run_summaries = runSqlJson(
    `WITH canonical_scores AS (
       SELECT
         CASE
           WHEN model_name = 'glm_logit' THEN 'glm_ridge'
           ELSE model_name
         END AS model_name,
         model_run_id,
         game_date_utc,
         log_loss,
         brier,
         accuracy
       FROM model_scores
       WHERE model_run_id IS NOT NULL
         AND TRIM(model_run_id) <> ''
     ),
     canonical_runs AS (
       SELECT
         model_run_id,
         CASE
           WHEN model_name = 'glm_logit' THEN 'glm_ridge'
           ELSE model_name
         END AS model_name,
         run_type,
         created_at_utc,
         snapshot_id,
         feature_set_version
       FROM model_runs
     ),
     summarized AS (
       SELECT
         cs.model_name,
         cs.model_run_id,
         cr.run_type,
         cr.created_at_utc,
         cr.snapshot_id,
         cr.feature_set_version,
         MIN(cs.game_date_utc) AS first_game_date_utc,
         MAX(cs.game_date_utc) AS last_game_date_utc,
         COUNT(*) AS n_games,
         AVG(cs.log_loss) AS avg_log_loss,
         AVG(cs.brier) AS avg_brier,
         AVG(cs.accuracy) AS accuracy
       FROM canonical_scores cs
       LEFT JOIN canonical_runs cr
         ON cr.model_run_id = cs.model_run_id
       GROUP BY
         cs.model_name,
         cs.model_run_id,
         cr.run_type,
         cr.created_at_utc,
         cr.snapshot_id,
         cr.feature_set_version
     ),
     ranked AS (
       SELECT
         *,
         ROW_NUMBER() OVER (
           PARTITION BY model_name
           ORDER BY
             DATETIME(COALESCE(created_at_utc, last_game_date_utc || 'T23:59:59Z')) DESC,
             model_run_id DESC
         ) AS version_rank
       FROM summarized
     )
     SELECT
       model_name,
       model_run_id,
       run_type,
       created_at_utc,
       snapshot_id,
       feature_set_version,
       first_game_date_utc,
       last_game_date_utc,
       n_games,
       avg_log_loss,
       avg_brier,
       accuracy,
       version_rank,
       CASE WHEN version_rank = 1 THEN 1 ELSE 0 END AS is_latest_version
     FROM ranked
     ORDER BY
       DATETIME(COALESCE(created_at_utc, last_game_date_utc || 'T23:59:59Z')) DESC,
       model_name ASC,
       avg_log_loss ASC
     LIMIT 240`,
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

  return NextResponse.json({ league, scores, run_summaries, change_points });
}
