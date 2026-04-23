"use client";

import { Suspense } from "react";
import NestedTournamentView from "@/components/NestedTournamentView";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { NestedTournamentResponse } from "@/lib/types";

const EMPTY_NESTED: NestedTournamentResponse = {
  league: "MLB",
  current_best: null,
  summary: null,
  target_coverage: [],
  family_champions: [],
  inter_family_leaderboard: [],
  artifacts: {},
};

function NestedTournamentPageContent() {
  const league = useLeague();
  const { data, isLoading, error } = useDashboardData<NestedTournamentResponse>(
    "nestedTournament",
    "/api/nested-tournament",
    league,
    EMPTY_NESTED
  );

  if (error) return <div className="card">{error}</div>;
  if (isLoading) return <p className="small">Loading nested tournament...</p>;

  return <NestedTournamentView data={data} />;
}

export default function NestedTournamentPage() {
  return (
    <Suspense fallback={<p className="small">Loading nested tournament...</p>}>
      <NestedTournamentPageContent />
    </Suspense>
  );
}

