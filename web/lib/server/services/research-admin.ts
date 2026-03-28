import { runSqlJson } from "@/lib/db";
import { type LeagueCode } from "@/lib/league";
import { getLatestUpcomingAsOf } from "@/lib/server/repositories/forecasts";
import { getActiveChampionSummary } from "@/lib/server/services/betting-driver";
import type {
  ResearchAdminResponse,
  ResearchBriefRow,
  ResearchChampionSummary,
  ResearchPromotionSummary,
  ResearchRunRow,
  TableRow,
} from "@/lib/types";

type RawBriefRow = {
  brief_id?: string | null;
  league?: string | null;
  profile_key?: string | null;
  brief_key?: string | null;
  title?: string | null;
  status?: string | null;
  source_path?: string | null;
  updated_at_utc?: string | null;
  brief_json?: string | null;
};

type RawRunRow = {
  run_id?: string | null;
  league?: string | null;
  profile_key?: string | null;
  brief_id?: string | null;
  brief_key?: string | null;
  incumbent_model_name?: string | null;
  candidate_model_name?: string | null;
  status?: string | null;
  auto_promote?: number | null;
  report_slug?: string | null;
  report_path?: string | null;
  scorecard_path?: string | null;
  fold_metrics_path?: string | null;
  promotion_path?: string | null;
  started_at_utc?: string | null;
  completed_at_utc?: string | null;
  summary_json?: string | null;
};

type RawDecisionRow = {
  promoted?: number | null;
  incumbent_model_name?: string | null;
  candidate_model_name?: string | null;
  reason_summary?: string | null;
  policy_json?: string | null;
  created_at_utc?: string | null;
};

function parseJsonRecord(value?: string | null): TableRow | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function listBriefs(league: LeagueCode): ResearchBriefRow[] {
  const rows = runSqlJson<RawBriefRow>(
    `SELECT
       brief_id,
       league,
       profile_key,
       brief_key,
       title,
       status,
       source_path,
       updated_at_utc,
       brief_json
     FROM experiment_briefs
     WHERE league = ${JSON.stringify(league)}
     ORDER BY updated_at_utc DESC, brief_key ASC`,
    { league }
  );
  return rows
    .filter((row) => row.brief_id && row.brief_key && row.title && row.updated_at_utc)
    .map((row) => ({
      brief_id: String(row.brief_id),
      league: String(row.league || league),
      profile_key: String(row.profile_key || "default"),
      brief_key: String(row.brief_key),
      title: String(row.title),
      status: String(row.status || "active"),
      source_path: row.source_path ? String(row.source_path) : null,
      updated_at_utc: String(row.updated_at_utc),
      brief: parseJsonRecord(row.brief_json),
    }));
}

function listRuns(league: LeagueCode): ResearchRunRow[] {
  const rows = runSqlJson<RawRunRow>(
    `SELECT
       run_id,
       league,
       profile_key,
       brief_id,
       brief_key,
       incumbent_model_name,
       candidate_model_name,
       status,
       auto_promote,
       report_slug,
       report_path,
       scorecard_path,
       fold_metrics_path,
       promotion_path,
       started_at_utc,
       completed_at_utc,
       summary_json
     FROM experiment_runs
     WHERE league = ${JSON.stringify(league)}
     ORDER BY started_at_utc DESC, run_id DESC`,
    { league }
  );
  return rows
    .filter((row) => row.run_id && row.status && row.started_at_utc)
    .map((row) => ({
      run_id: String(row.run_id),
      league: String(row.league || league),
      profile_key: String(row.profile_key || "default"),
      brief_id: row.brief_id ? String(row.brief_id) : null,
      brief_key: row.brief_key ? String(row.brief_key) : null,
      incumbent_model_name: row.incumbent_model_name ? String(row.incumbent_model_name) : null,
      candidate_model_name: row.candidate_model_name ? String(row.candidate_model_name) : null,
      status: String(row.status),
      auto_promote: Number(row.auto_promote || 0),
      report_slug: row.report_slug ? String(row.report_slug) : null,
      report_path: row.report_path ? String(row.report_path) : null,
      scorecard_path: row.scorecard_path ? String(row.scorecard_path) : null,
      fold_metrics_path: row.fold_metrics_path ? String(row.fold_metrics_path) : null,
      promotion_path: row.promotion_path ? String(row.promotion_path) : null,
      started_at_utc: String(row.started_at_utc),
      completed_at_utc: row.completed_at_utc ? String(row.completed_at_utc) : null,
      summary: parseJsonRecord(row.summary_json),
    }));
}

function listDecisions(league: LeagueCode): ResearchPromotionSummary[] {
  const rows = runSqlJson<RawDecisionRow>(
    `SELECT
       promoted,
       incumbent_model_name,
       candidate_model_name,
       reason_summary,
       policy_json,
       created_at_utc
     FROM promotion_decisions
     WHERE league = ${JSON.stringify(league)}
     ORDER BY created_at_utc DESC, decision_id DESC`,
    { league }
  );
  return rows.map((row) => ({
    promoted: Boolean(Number(row.promoted || 0)),
    incumbent_model_name: row.incumbent_model_name ? String(row.incumbent_model_name) : null,
    candidate_model_name: row.candidate_model_name ? String(row.candidate_model_name) : null,
    reason_summary: row.reason_summary ? String(row.reason_summary) : null,
    policy: parseJsonRecord(row.policy_json),
    created_at_utc: row.created_at_utc ? String(row.created_at_utc) : null,
  }));
}

export async function getResearchAdminPayload(league: LeagueCode): Promise<ResearchAdminResponse> {
  const champion = getActiveChampionSummary(league);
  return {
    league,
    as_of_utc: getLatestUpcomingAsOf(league),
    champion: (champion as ResearchChampionSummary | null) ?? null,
    briefs: listBriefs(league),
    runs: listRuns(league),
    decisions: listDecisions(league),
  };
}
