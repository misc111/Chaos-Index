"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague, withLeague } from "@/lib/league";

function CalibrationPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetch(withLeague("/api/metrics", league), { cache: "no-store" })
      .then((r) => r.json())
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
