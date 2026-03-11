"use client";

import { Suspense } from "react";
import EnsembleSnapshotExplorer from "@/components/EnsembleSnapshotExplorer";
import ModelBetReplayExplorer from "@/components/ModelBetReplayExplorer";
import ModelTable from "@/components/ModelTable";
import ModelVersionExplorer from "@/components/ModelVersionExplorer";
import PerformanceCharts from "@/components/PerformanceCharts";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
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
      />
      <div className="card">
        <h3 className="title">Model Drift Timeline</h3>
        <p className="small">
          The top bankroll chart freezes each dated ensemble snapshot into its own betting path, so you can answer the direct
          counterfactual first: what your account would look like today if you had stopped recalibrating on a given date. The score
          charts underneath then show how each model family has been grading over time, while the version replay section keeps older
          trained runs separate so you can inspect which feature-set or parameter changes lined up with better or worse live results.
        </p>
      </div>
      <PerformanceCharts rows={data.scores} />
      <ModelVersionExplorer rows={data.run_summaries} />
      <ModelBetReplayExplorer
        runs={data.replay_runs}
        defaultStrategy={data.default_replay_strategy}
        comparisonStrategy={data.comparison_replay_strategy}
      />
      <ModelTable title="Change-Point Alerts" rows={data.change_points} />
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
