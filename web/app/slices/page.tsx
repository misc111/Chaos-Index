"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague, withLeague } from "@/lib/league";

function SlicesPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetch(withLeague("/api/metrics", league), { cache: "no-store" })
      .then((r) => r.json())
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
