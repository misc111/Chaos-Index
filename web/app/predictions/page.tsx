"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import Filters from "@/components/Filters";
import { ForecastRow } from "@/lib/types";
import { normalizeLeague, withLeague } from "@/lib/league";

function PredictionsPageContent() {
  const [rows, setRows] = useState<ForecastRow[]>([]);
  const [team, setTeam] = useState("");
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetch(withLeague("/api/predictions", league), { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setRows(d.rows || []));
  }, [league]);

  const teams = useMemo(
    () => Array.from(new Set(rows.flatMap((r) => [r.home_team, r.away_team]))).sort(),
    [rows]
  );

  const filtered = rows.filter((r) => !team || r.home_team === team || r.away_team === team);

  return (
    <div className="grid">
      <Filters teams={teams} value={team} onChange={setTeam} />
      <div className="card" style={{ overflowX: "auto" }}>
        <h2 className="title">Upcoming Games</h2>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Matchup</th>
              <th>Ensemble Home Win</th>
              <th>Winner</th>
              <th>Spread (mean +/- sd)</th>
              <th>Bayes CI</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.game_id}>
                <td>{r.game_date_utc}</td>
                <td>
                  {r.away_team} @ {r.home_team}
                </td>
                <td>{(r.ensemble_prob_home_win * 100).toFixed(1)}%</td>
                <td>
                  <span className="badge">{r.predicted_winner}</span>
                </td>
                <td>
                  {r.spread_mean?.toFixed(3)} +/- {r.spread_sd?.toFixed(3)}
                </td>
                <td>
                  {r.bayes_ci_low?.toFixed(3)} - {r.bayes_ci_high?.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function PredictionsPage() {
  return (
    <Suspense fallback={<p className="small">Loading predictions...</p>}>
      <PredictionsPageContent />
    </Suspense>
  );
}
