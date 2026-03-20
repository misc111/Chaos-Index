"use client";

import { useSearchParams } from "next/navigation";
import { getDefaultBetStrategyForLeague, normalizeBetStrategy } from "@/lib/betting-strategy";
import type { LeagueCode } from "@/lib/league";

export function useBetStrategy(league?: LeagueCode | null) {
  const searchParams = useSearchParams();
  const strategy = searchParams.get("strategy");
  return strategy ? normalizeBetStrategy(strategy) : getDefaultBetStrategyForLeague(league);
}
