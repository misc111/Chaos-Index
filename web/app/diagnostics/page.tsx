"use client";

import { Suspense } from "react";
import ModelTable from "@/components/ModelTable";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { ValidationResponse } from "@/lib/types";

const EMPTY_VALIDATION: ValidationResponse = {
  sections: {},
};

function DiagnosticsPageContent() {
  const league = useLeague();
  const { data, isLoading, error } = useDashboardData<ValidationResponse>(
    "validation",
    "/api/validation",
    league,
    EMPTY_VALIDATION
  );

  if (error) {
    return <div className="card">{error}</div>;
  }
  if (isLoading) {
    return <p className="small">Loading diagnostics...</p>;
  }

  return <ModelTable title="GLM/ML Diagnostics Snapshot" rows={data.significance || []} />;
}

export default function DiagnosticsPage() {
  return (
    <Suspense fallback={<p className="small">Loading diagnostics...</p>}>
      <DiagnosticsPageContent />
    </Suspense>
  );
}
