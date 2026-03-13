"use client";

import { useSearchParams } from "next/navigation";
import { getPerformanceReplayExperimentSummary } from "@/lib/performance-replay-experiments";

export function usePerformanceReplayExperiment() {
  const searchParams = useSearchParams();
  const rawExperiment = searchParams.get("experiment");
  if (!rawExperiment || rawExperiment === "baseline") {
    return null;
  }

  return getPerformanceReplayExperimentSummary(rawExperiment);
}
