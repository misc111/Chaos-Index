"use client";

import { useEffect, useState } from "react";
import ModelTable from "@/components/ModelTable";

export default function CalibrationPage() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);

  useEffect(() => {
    fetch("/api/metrics", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setRows(d.calibration || []));
  }, []);

  return <ModelTable title="Calibration Metrics (alpha/beta/ECE/MCE)" rows={rows} />;
}
