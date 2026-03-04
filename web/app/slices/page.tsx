"use client";

import { useEffect, useState } from "react";
import ModelTable from "@/components/ModelTable";

export default function SlicesPage() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);

  useEffect(() => {
    fetch("/api/metrics", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setRows(d.slices || []));
  }, []);

  return <ModelTable title="Slice Analysis + Drift" rows={rows} />;
}
