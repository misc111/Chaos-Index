"use client";

import { Suspense } from "react";
import ValidationTabs from "@/components/ValidationTabs";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { ValidationResponse } from "@/lib/types";

const EMPTY_VALIDATION: ValidationResponse = {
  sections: {},
};

function ValidationPageContent() {
  const league = useLeague();
  const { data, isLoading, error } = useDashboardData<ValidationResponse>("validation", "/api/validation", league, EMPTY_VALIDATION);

  if (error) {
    return <div className="card">{error}</div>;
  }
  if (isLoading) {
    return <p className="small">Loading validation...</p>;
  }

  return <ValidationTabs sections={data.sections} />;
}

export default function ValidationPage() {
  return (
    <Suspense fallback={<p className="small">Loading validation...</p>}>
      <ValidationPageContent />
    </Suspense>
  );
}
