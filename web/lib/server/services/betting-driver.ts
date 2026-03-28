import type { BetRiskRegime } from "@/lib/betting-strategy";
import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";

const DEFAULT_BETTING_MODEL = "ensemble";
const NBA_BETTING_MODEL = "glm_elastic_net";
const DEFAULT_PROFILE_KEY = "default";
type BettingDriverMode = "live" | "historicalReplay";
const NBA_GUARDED_LOOKBACK_DAYS = 7;
const NBA_GUARDED_MIN_MODELS = 3;
const NBA_GUARDED_MODELS = new Set([
  "ensemble",
  "glm_elastic_net",
  "glm_lasso",
  "glm_ridge",
  "dynamic_rating",
  "bayes_bt_state_space",
]);

type RawChangePointRow = {
  model_name?: string | null;
  detected_date?: string | null;
};

type RawActiveChampionRow = {
  model_name?: string | null;
  promoted_at_utc?: string | null;
  source_run_id?: string | null;
  source_brief_id?: string | null;
  descriptor_json?: string | null;
  policy_json?: string | null;
};

export type ActiveChampionSummary = {
  league: LeagueCode;
  profile_key: string;
  model_name: string;
  promoted_at_utc?: string | null;
  source_run_id?: string | null;
  source_brief_id?: string | null;
  descriptor?: Record<string, unknown> | null;
  policy?: Record<string, unknown> | null;
};

export type BettingDriverContext = {
  league: LeagueCode;
  profile_key: string;
  mode: BettingDriverMode;
  preferred_model_name: string;
  champion: ActiveChampionSummary | null;
  desk_posture: BetRiskRegime;
};

function subtractDays(dateKey: string, days: number): string {
  const base = new Date(`${dateKey}T12:00:00Z`);
  base.setUTCDate(base.getUTCDate() - days);
  return base.toISOString().slice(0, 10);
}

function parseJsonRecord(value?: string | null): Record<string, unknown> | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {}
  return null;
}

export function getActiveChampionSummary(
  league: LeagueCode,
  profileKey = DEFAULT_PROFILE_KEY
): ActiveChampionSummary | null {
  const rows = runSqlJson<RawActiveChampionRow>(
    `
    SELECT
      model_name,
      promoted_at_utc,
      source_run_id,
      source_brief_id,
      descriptor_json,
      policy_json
    FROM active_champions
    WHERE league = '${league}'
      AND profile_key = '${profileKey}'
    LIMIT 1
    `,
    { league }
  );
  const row = rows[0];
  const modelName = String(row?.model_name || "").trim();
  if (!modelName) {
    return null;
  }
  return {
    league,
    profile_key: profileKey,
    model_name: modelName,
    promoted_at_utc: row?.promoted_at_utc ?? null,
    source_run_id: row?.source_run_id ?? null,
    source_brief_id: row?.source_brief_id ?? null,
    descriptor: parseJsonRecord(row?.descriptor_json),
    policy: parseJsonRecord(row?.policy_json),
  };
}

export function getPreferredBettingModelName(
  league: LeagueCode,
  mode: BettingDriverMode = "live"
): string {
  return getBettingDriverContext(league, mode).preferred_model_name;
}

export function getActiveBetRiskRegime(league: LeagueCode): BetRiskRegime {
  if (league !== "NBA") {
    return "normal";
  }

  const rows = runSqlJson<RawChangePointRow>(
    `WITH latest AS (
       SELECT MAX(as_of_utc) AS as_of_utc
       FROM change_points
     )
     SELECT
       model_name,
       json_extract(details_json, '$.date') AS detected_date
     FROM change_points
     WHERE as_of_utc = (SELECT as_of_utc FROM latest)`,
    { league }
  );

  const latestDate = rows
    .map((row) => String(row.detected_date || "").trim())
    .filter((value) => /^\d{4}-\d{2}-\d{2}$/.test(value))
    .sort()
    .at(-1);
  if (!latestDate) {
    return "normal";
  }

  const cutoffDate = subtractDays(latestDate, NBA_GUARDED_LOOKBACK_DAYS - 1);
  const impactedModels = new Set(
    rows
      .filter((row) => {
        const modelName = String(row.model_name || "").trim();
        const detectedDate = String(row.detected_date || "").trim();
        return NBA_GUARDED_MODELS.has(modelName) && detectedDate >= cutoffDate;
      })
      .map((row) => String(row.model_name || "").trim())
  );

  return impactedModels.size >= NBA_GUARDED_MIN_MODELS ? "guarded" : "normal";
}

export function getBettingDriverContext(
  league: LeagueCode,
  mode: BettingDriverMode = "live",
  profileKey = DEFAULT_PROFILE_KEY
): BettingDriverContext {
  const champion = getActiveChampionSummary(league, profileKey);
  return {
    league,
    profile_key: profileKey,
    mode,
    preferred_model_name: champion?.model_name || (league === "NBA" ? NBA_BETTING_MODEL : DEFAULT_BETTING_MODEL),
    champion,
    desk_posture: getActiveBetRiskRegime(league),
  };
}
