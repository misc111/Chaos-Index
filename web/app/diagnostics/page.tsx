"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague, withLeague } from "@/lib/league";

function DiagnosticsPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetch(withLeague("/api/validation", league), { cache: "no-store" })
      .then((r) => r.json())
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
