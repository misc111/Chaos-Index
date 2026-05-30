import { runSqlJson } from "@/lib/db";
import type { LeagueCode } from "@/lib/league";
import { getActiveChampionSummary } from "@/lib/server/services/betting-driver";
export { loadEvidenceStatus } from "@/lib/server/research-evidence-status";
import type {
  ResearchChampionSummary,
  ResearchDeskNightlyRow,
  ResearchDeskResponse,
  ResearchPromotionSummary,
  TableRow,
} from "@/lib/types";
import type { BetDecision } from "@/lib/betting";

type NightlyBoardInput = {
  scheduled_game_count?: number;
  fresh_forecast_status?: "available" | "missing" | "no_slate" | string;
  rows: Array<{
    game_id: number;
    start_time_utc?: string | null;
    home_team_name: string;
    away_team_name: string;
    home_win_probability: number;
    betting_model_name?: string | null;
  }>;
};

type RawPromotionRow = {
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

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function displayModelName(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) return "candidate";
  return raw
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((token) => {
      const lower = token.toLowerCase();
      if (["glm", "gam", "dglm", "mars"].includes(lower)) return lower.toUpperCase();
      return token.charAt(0).toUpperCase() + token.slice(1);
    })
    .join(" ");
}

function summarizePromotion(promotion: ResearchPromotionSummary | null): string {
  if (!promotion) return "No promotion review yet.";
  const candidate = displayModelName(promotion.candidate_model_name);
  return promotion.promoted ? `${candidate} became the approved model.` : `${candidate} was tested but was not approved.`;
}

export function buildUnsupportedPayload(league: LeagueCode): ResearchDeskResponse {
  const emptySummary = buildOvernightSummary({
    league,
    deskPosture: "normal",
    championModelName: null,
    promotion: null,
    totalGames: 0,
    betCount: 0,
  });

  return {
    league,
    as_of_utc: null,
    odds_as_of_utc: null,
    date_central: undefined,
    scheduled_game_count: 0,
    fresh_forecast_status: "no_slate",
    desk_posture: "normal",
    overnight_summary: emptySummary,
    champion: null,
    model_diagnostics: {},
    latest_promotion: null,
    source_kind: null,
    source_status: null,
    evidence_stage: null,
    latest_artifact_role: null,
    promotion_eligible: false,
    production_ready: false,
    evidence_status: null,
    current_best_top_models: [],
    counts: {
      total_games: 0,
      bets: 0,
      passes: 0,
    },
    rows: [],
  };
}

export function loadLatestPromotion(league: LeagueCode): ResearchPromotionSummary | null {
  try {
    const rows = runSqlJson<RawPromotionRow>(
      `SELECT
         promoted,
         incumbent_model_name,
         candidate_model_name,
         reason_summary,
         policy_json,
         created_at_utc
       FROM promotion_decisions
       WHERE league = ${JSON.stringify(league)}
         AND profile_key = 'default'
       ORDER BY created_at_utc DESC, decision_id DESC
       LIMIT 1`,
      { league }
    );
    const row = rows[0];
    if (!row) return null;
    return {
      promoted: Boolean(Number(row.promoted || 0)),
      incumbent_model_name: row.incumbent_model_name ? String(row.incumbent_model_name) : null,
      candidate_model_name: row.candidate_model_name ? String(row.candidate_model_name) : null,
      reason_summary: row.reason_summary ? String(row.reason_summary) : null,
      policy: parseJsonRecord(row.policy_json),
      created_at_utc: row.created_at_utc ? String(row.created_at_utc) : null,
    };
  } catch {
    return null;
  }
}

export function loadChampionSummary(league: LeagueCode): ResearchChampionSummary | null {
  const champion = getActiveChampionSummary(league);
  if (!champion?.model_name || !champion.promoted_at_utc) {
    return null;
  }
  return {
    league: champion.league,
    profile_key: champion.profile_key,
    model_name: champion.model_name,
    promoted_at_utc: champion.promoted_at_utc,
    source_run_id: champion.source_run_id ?? null,
    source_brief_id: champion.source_brief_id ?? null,
    descriptor: champion.descriptor ?? null,
    policy: champion.policy ?? null,
  };
}

export function buildNightlyRows(
  marketBoard: NightlyBoardInput,
  decisions: BetDecision[]
): ResearchDeskNightlyRow[] {
  return marketBoard.rows.map((row, index) => {
    const decision = decisions[index];
    const isBet = Boolean(decision && decision.stake > 0 && decision.side !== "none");
    return {
      game_id: row.game_id,
      start_time_utc: row.start_time_utc,
      home_team: row.home_team_name,
      away_team: row.away_team_name,
      bet_label: isBet ? "bet" : "pass",
      reason: decision?.reason || "No recommendation available",
      side: decision?.side || "none",
      team: decision?.team || null,
      stake: decision?.stake || 0,
      odds: decision?.odds ?? null,
      edge: decision?.edge ?? null,
      expected_value: decision?.expectedValue ?? null,
      home_win_probability: row.home_win_probability,
      betting_model_name: row.betting_model_name ?? null,
    };
  });
}

export function buildOvernightSummary(args: {
  league: LeagueCode;
  deskPosture: "normal" | "guarded";
  championModelName?: string | null;
  promotion: ResearchPromotionSummary | null;
  totalGames: number;
  betCount: number;
  scheduledGameCount?: number;
  freshForecastStatus?: string | null;
}): string {
  const riskLabel = args.deskPosture === "guarded" ? "more cautious than usual" : "normal";
  const modelSentence = args.championModelName
    ? `Approved model: ${displayModelName(args.championModelName)}.`
    : "No model is fully approved yet.";
  const scheduledGameCount = Number.isFinite(Number(args.scheduledGameCount))
    ? Math.max(0, Number(args.scheduledGameCount))
    : 0;
  const slateLead =
    args.totalGames > 0
      ? `Today: ${pluralize(args.totalGames, "game")} reviewed, ${pluralize(args.betCount, "suggested bet")}, ${pluralize(
          Math.max(0, args.totalGames - args.betCount),
          "no-bet game"
        )}.`
      : scheduledGameCount > 0 && args.freshForecastStatus === "missing"
        ? `Today: ${pluralize(scheduledGameCount, "game")} scheduled; no fresh forecast snapshot is loaded yet.`
      : "No games are loaded for today yet.";

  return [slateLead, `Risk level: ${riskLabel}.`, modelSentence, summarizePromotion(args.promotion)].join(" ");
}
