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

function CalibrationPageContent() {
  const league = useLeague();
  const { data, isLoading, error } = useDashboardData<MetricsResponse>("metrics", "/api/metrics", league, EMPTY_METRICS);

  if (error) {
    return <div className="card">{error}</div>;
  }
  if (isLoading) {
    return <p className="small">Loading calibration...</p>;
  }

  return <ModelTable title="Calibration Metrics (alpha/beta/ECE/MCE)" rows={data.calibration} />;
}

export default function CalibrationPage() {
  return (
    <Suspense fallback={<p className="small">Loading calibration...</p>}>
      <CalibrationPageContent />
    </Suspense>
  );
}
