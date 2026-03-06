"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";

function LeaderboardPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetchDashboardJson<{ leaderboard?: Record<string, any>[] }>("metrics", "/api/metrics", league)
      .then((d) => setRows(d.leaderboard || []));
  }, [league]);

  return <ModelTable title="Leaderboard (rolling + cumulative)" rows={rows} />;
}

export default function LeaderboardPage() {
  return (
    <Suspense fallback={<p className="small">Loading leaderboard...</p>}>
      <LeaderboardPageContent />
    </Suspense>
  );
}
