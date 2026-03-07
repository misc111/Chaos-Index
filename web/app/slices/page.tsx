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

function SlicesPageContent() {
  const league = useLeague();
  const { data, isLoading, error } = useDashboardData<MetricsResponse>("metrics", "/api/metrics", league, EMPTY_METRICS);

  if (error) {
    return <div className="card">{error}</div>;
  }
  if (isLoading) {
    return <p className="small">Loading slices...</p>;
  }

  return <ModelTable title="Slice Analysis + Drift" rows={data.slices} />;
}

export default function SlicesPage() {
  return (
    <Suspense fallback={<p className="small">Loading slices...</p>}>
      <SlicesPageContent />
    </Suspense>
  );
}
