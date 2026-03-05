"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ModelTable from "@/components/ModelTable";
import { normalizeLeague, withLeague } from "@/lib/league";

function LeaderboardPageContent() {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetch(withLeague("/api/metrics", league), { cache: "no-store" })
      .then((r) => r.json())
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
