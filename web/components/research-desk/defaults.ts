import type { ResearchDeskResponse } from "@/lib/types";

export const EMPTY_RESEARCH_DESK: ResearchDeskResponse = {
  league: "MLB",
  as_of_utc: null,
  odds_as_of_utc: null,
  date_central: undefined,
  desk_posture: "normal",
  overnight_summary: null,
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
  counts: { total_games: 0, bets: 0, passes: 0 },
  rows: [],
};
