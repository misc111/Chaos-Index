"use client";

import { Suspense } from "react";
import ModelTable from "@/components/ModelTable";
import PerformanceCharts from "@/components/PerformanceCharts";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { PerformanceResponse } from "@/lib/types";

const EMPTY_PERFORMANCE: PerformanceResponse = {
  league: "NHL",
  scores: [],
  change_points: [],
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
      <PerformanceCharts rows={data.scores} />
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
