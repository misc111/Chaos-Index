import type { LeagueCode } from "@/lib/league";

const DEFAULT_BETTING_MODEL = "ensemble";
type BettingDriverMode = "live" | "historicalReplay";

export function getPreferredBettingModelName(
  league: LeagueCode,
  mode: BettingDriverMode = "live"
): string {
  void league;
  void mode;
  return DEFAULT_BETTING_MODEL;
}
