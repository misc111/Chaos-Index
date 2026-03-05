"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import PerformanceCharts from "@/components/PerformanceCharts";
import { normalizeLeague, withLeague } from "@/lib/league";

function PerformancePageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [alerts, setAlerts] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetch(withLeague("/api/performance", league), { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        setRows(d.scores || []);
        setAlerts(d.change_points || []);
      });
  }, [league]);

  return (
    <div className="grid">
      <PerformanceCharts rows={rows as any} />
      <ModelTable title="Change-Point Alerts" rows={alerts} />
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
