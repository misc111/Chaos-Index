"use client";

import { Suspense } from "react";
import ModelTable from "@/components/ModelTable";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { MetricsResponse } from "@/lib/types";

const EMPTY_METRICS: MetricsResponse = {
  leaderboard: [],
  calibration: [],
  slices: [],
};

function LeaderboardPageContent() {
  const league = useLeague();
  const { data, isLoading, error } = useDashboardData<MetricsResponse>("metrics", "/api/metrics", league, EMPTY_METRICS);

  if (error) {
    return <div className="card">{error}</div>;
  }
  if (isLoading) {
    return <p className="small">Loading leaderboard...</p>;
  }

  return <ModelTable title="Leaderboard (rolling + cumulative)" rows={data.leaderboard} />;
}

export default function LeaderboardPage() {
  return (
    <Suspense fallback={<p className="small">Loading leaderboard...</p>}>
      <LeaderboardPageContent />
    </Suspense>
  );
}
