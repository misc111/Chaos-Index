import type { LeagueCode } from "@/lib/league";

const DEFAULT_BETTING_MODEL = "ensemble";
const NBA_BETTING_MODEL = "glm_elastic_net";
type BettingDriverMode = "live" | "historicalReplay";

export function getPreferredBettingModelName(
  league: LeagueCode,
  mode: BettingDriverMode = "live"
): string {
  void mode;
  return league === "NBA" ? NBA_BETTING_MODEL : DEFAULT_BETTING_MODEL;
}
