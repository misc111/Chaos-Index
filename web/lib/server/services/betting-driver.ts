import type { BetRiskRegime } from "@/lib/betting-strategy";
import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";

const DEFAULT_BETTING_MODEL = "ensemble";
const DEFAULT_PROFILE_KEY = "default";
type BettingDriverMode = "live" | "historicalReplay";

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
  return "normal";
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
    preferred_model_name: champion?.model_name || DEFAULT_BETTING_MODEL,
    champion,
    desk_posture: getActiveBetRiskRegime(league),
  };
}
