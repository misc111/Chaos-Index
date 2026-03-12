"use client";

import { Suspense } from "react";
import EnsembleSnapshotExplorer from "@/components/EnsembleSnapshotExplorer";
import ModelBetReplayExplorer from "@/components/ModelBetReplayExplorer";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useBetStrategy } from "@/lib/hooks/useBetStrategy";
import { useLeague } from "@/lib/hooks/useLeague";
import { usePerformanceReplayExperiment } from "@/lib/hooks/usePerformanceReplayExperiment";
import type { PerformanceResponse } from "@/lib/types";

const EMPTY_PERFORMANCE: PerformanceResponse = {
  league: "NHL",
  scores: [],
  run_summaries: [],
  change_points: [],
  replay_runs: [],
  ensemble_snapshots: [],
  replay_experiment: null,
};

function PerformancePageContent() {
  const league = useLeague();
  const strategy = useBetStrategy();
  const replayExperiment = usePerformanceReplayExperiment();
  const livePath = replayExperiment ? `/api/performance?experiment=${encodeURIComponent(replayExperiment.id)}` : "/api/performance";
  const { data, isLoading, error } = useDashboardData<PerformanceResponse>(
    "performance",
    livePath,
    league,
    EMPTY_PERFORMANCE,
    undefined,
    replayExperiment?.id
  );

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
        replayExperiment={data.replay_experiment}
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
