"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";

function DiagnosticsPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetchDashboardJson<{ significance?: Record<string, any>[] }>("validation", "/api/validation", league)
      .then((d) => setRows(d.significance || []));
  }, [league]);

  return <ModelTable title="GLM/ML Diagnostics Snapshot" rows={rows} />;
}

export default function DiagnosticsPage() {
  return (
    <Suspense fallback={<p className="small">Loading diagnostics...</p>}>
      <DiagnosticsPageContent />
    </Suspense>
  );
}
