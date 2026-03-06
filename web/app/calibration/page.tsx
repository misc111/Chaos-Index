"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";

function CalibrationPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetchDashboardJson<{ calibration?: Record<string, any>[] }>("metrics", "/api/metrics", league)
      .then((d) => setRows(d.calibration || []));
  }, [league]);

  return <ModelTable title="Calibration Metrics (alpha/beta/ECE/MCE)" rows={rows} />;
}

export default function CalibrationPage() {
  return (
    <Suspense fallback={<p className="small">Loading calibration...</p>}>
      <CalibrationPageContent />
    </Suspense>
  );
}
