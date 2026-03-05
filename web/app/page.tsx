"use client";

import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { normalizeLeague, withLeague } from "@/lib/league";

function HomePageContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  return (
    <div className="grid two">
      <div className="card">
        <h2 className="title">Pipeline</h2>
        <p className="small">
          Daily workflow: fetch {"->"} features {"->"} train/update {"->"} predict {"->"} ingest results {"->"} score {"->"} performance/validation artifacts.
        </p>
        <p className="small">Use Make targets from repo root to run each stage.</p>
      </div>
      <div className="card">
        <h2 className="title">Quick Links</h2>
        <p>
          <Link href={withLeague("/predictions", league)}>Upcoming forecasts</Link>
        </p>
        <p>
          <Link href={withLeague("/actual-vs-expected", league)}>Actual vs Expected</Link>
        </p>
        <p>
          <Link href={withLeague("/leaderboard", league)}>Model leaderboard</Link>
        </p>
        <p>
          <Link href={withLeague("/validation", league)}>Validation suite</Link>
        </p>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={<p className="small">Loading...</p>}>
      <HomePageContent />
    </Suspense>
  );
}
