"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";

function SlicesPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetchDashboardJson<{ slices?: Record<string, any>[] }>("metrics", "/api/metrics", league)
      .then((d) => setRows(d.slices || []));
  }, [league]);

  return <ModelTable title="Slice Analysis + Drift" rows={rows} />;
}

export default function SlicesPage() {
  return (
    <Suspense fallback={<p className="small">Loading slices...</p>}>
      <SlicesPageContent />
    </Suspense>
  );
}
