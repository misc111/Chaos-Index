"use client";

import { useSearchParams } from "next/navigation";
import { getPerformanceReplayExperimentSummary } from "@/lib/performance-replay-experiments";

export function usePerformanceReplayExperiment() {
  const searchParams = useSearchParams();
  return getPerformanceReplayExperimentSummary(searchParams.get("experiment"));
}
