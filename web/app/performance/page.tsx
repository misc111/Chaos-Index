"use client";

import { Suspense } from "react";
import EnsembleSnapshotExplorer from "@/components/EnsembleSnapshotExplorer";
import ModelBetReplayExplorer from "@/components/ModelBetReplayExplorer";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useBetStrategy } from "@/lib/hooks/useBetStrategy";
import { useLeague } from "@/lib/hooks/useLeague";
import type { PerformanceResponse } from "@/lib/types";

const EMPTY_PERFORMANCE: PerformanceResponse = {
  league: "NHL",
  scores: [],
  run_summaries: [],
  change_points: [],
  replay_runs: [],
  ensemble_snapshots: [],
};

function PerformancePageContent() {
  const league = useLeague();
  const strategy = useBetStrategy();
  const { data, isLoading, error } = useDashboardData<PerformanceResponse>("performance", "/api/performance", league, EMPTY_PERFORMANCE);

  if (error) {
    return <div className="card">{error}</div>;
  }
  if (isLoading) {
    return <p className="small">Loading performance...</p>;
  }

  return (
    <div className="grid">
      <EnsembleSnapshotExplorer
        snapshots={data.ensemble_snapshots}
        defaultStrategy={data.default_replay_strategy}
        comparisonStrategy={data.comparison_replay_strategy}
        activeStrategy={strategy}
      />
      <ModelBetReplayExplorer
        runs={data.replay_runs}
        defaultStrategy={data.default_replay_strategy}
        comparisonStrategy={data.comparison_replay_strategy}
      />
    </div>
  );
}

export default function PerformancePage() {
  return (
    <Suspense fallback={<p className="small">Loading performance...</p>}>
      <PerformancePageContent />
    </Suspense>
  );
}
