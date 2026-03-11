"use client";

import { Suspense } from "react";
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
      <div className="card">
        <h3 className="title">Model Drift Timeline</h3>
        <p className="small">
          The top charts show how each model family has been scoring over time. The version replay section underneath keeps older
          trained runs separate, so you can see whether a feature-set change or parameter tweak coincided with worse live results.
          The new versioned bet replay block then asks the next question: what those dated model snapshots would actually have bet.
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
