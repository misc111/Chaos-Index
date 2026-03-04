"use client";

import { useEffect, useState } from "react";
import ModelTable from "@/components/ModelTable";
import PerformanceCharts from "@/components/PerformanceCharts";

export default function PerformancePage() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [alerts, setAlerts] = useState<Record<string, any>[]>([]);

  useEffect(() => {
    fetch("/api/performance", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        setRows(d.scores || []);
        setAlerts(d.change_points || []);
      });
  }, []);

  return (
    <div className="grid">
      <PerformanceCharts rows={rows as any} />
      <ModelTable title="Change-Point Alerts" rows={alerts} />
    </div>
  );
}
