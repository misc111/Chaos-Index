import type { BetRiskRegime } from "@/lib/betting-strategy";
import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";

const DEFAULT_BETTING_MODEL = "ensemble";
const NBA_BETTING_MODEL = "glm_elastic_net";
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

function subtractDays(dateKey: string, days: number): string {
  const base = new Date(`${dateKey}T12:00:00Z`);
  base.setUTCDate(base.getUTCDate() - days);
  return base.toISOString().slice(0, 10);
}

export function getPreferredBettingModelName(
  league: LeagueCode,
  mode: BettingDriverMode = "live"
): string {
  void mode;
  return league === "NBA" ? NBA_BETTING_MODEL : DEFAULT_BETTING_MODEL;
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
