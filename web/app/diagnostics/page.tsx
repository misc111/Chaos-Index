"use client";

import { useEffect, useState } from "react";
import ModelTable from "@/components/ModelTable";

export default function DiagnosticsPage() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);

  useEffect(() => {
    fetch("/api/validation", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setRows(d.significance || []));
  }, []);

  return <ModelTable title="GLM/ML Diagnostics Snapshot" rows={rows} />;
}
