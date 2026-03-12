"use client";

import { useSearchParams } from "next/navigation";
import { type LeagueCode } from "@/lib/league";
import {
  defaultPerformanceReplayExperimentForLeague,
  getPerformanceReplayExperimentSummary,
} from "@/lib/performance-replay-experiments";

export function usePerformanceReplayExperiment(league: LeagueCode) {
  const searchParams = useSearchParams();
  const rawExperiment = searchParams.get("experiment");
  if (rawExperiment === "baseline") {
    return null;
  }

  return getPerformanceReplayExperimentSummary(
    rawExperiment || defaultPerformanceReplayExperimentForLeague(league)
  );
}
