"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { normalizeLeague, withLeague } from "@/lib/league";
import styles from "./styles.module.css";

type GamesTodayRow = {
  game_id: number;
  game_date_utc?: string | null;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  start_time_utc?: string | null;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  home_moneyline_book?: string | null;
  away_moneyline_book?: string | null;
  over_190_price?: number | null;
  over_190_point?: number | null;
  over_190_book?: string | null;
};

type GamesTodayResponse = {
  league?: string;
  as_of_utc?: string | null;
  odds_as_of_utc?: string | null;
  date_central?: string;
  rows?: GamesTodayRow[];
};

type ExpectedSide = "home" | "away" | "none";

function expectedSide(homeWinProbability: number): ExpectedSide {
  if (homeWinProbability > 0.55) return "home";
  if (homeWinProbability < 0.45) return "away";
  return "none";
}

function expectedWinChance(homeWinProbability: number, side: ExpectedSide): number {
  if (side === "home") return homeWinProbability;
  if (side === "away") return 1 - homeWinProbability;
  return Math.max(homeWinProbability, 1 - homeWinProbability);
}

function formatAsOfLabel(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatMoneyline(value?: number | null): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric === 0) return "—";
  const rounded = Math.round(numeric);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

function formatOver190(value?: number | null, point?: number | null): string {
  const price = formatMoneyline(value);
  const p = Number(point);
  if (price === "—") return "—";
  if (!Number.isFinite(p)) return price;
  return `${price} @ ${p.toFixed(1)}`;
}

function GamesTodayPageContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const [rows, setRows] = useState<GamesTodayRow[]>([]);
  const [latestAsOf, setLatestAsOf] = useState<string>("");
  const [latestOddsAsOf, setLatestOddsAsOf] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetch(withLeague("/api/games-today", league), { cache: "no-store" })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Request failed: ${res.status}`);
        }
        return res.json() as Promise<GamesTodayResponse>;
      })
      .then((payload) => {
        if (cancelled) return;
        setRows(payload.rows || []);
        setLatestAsOf(typeof payload.as_of_utc === "string" ? payload.as_of_utc : "");
        setLatestOddsAsOf(typeof payload.odds_as_of_utc === "string" ? payload.odds_as_of_utc : "");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [league]);

  return (
    <div className="grid">
      <div className={`card ${styles.tableCard}`}>
        <h2 className="title">Games Today</h2>
        <p className="small">Only games scheduled for today (Central Time) are shown.</p>
        <p className="small">Anticipated winner threshold: win chance greater than 55%.</p>
        {latestAsOf ? <p className="small">Forecast snapshot as of {formatAsOfLabel(latestAsOf)}</p> : null}
        {latestOddsAsOf ? <p className="small">Odds snapshot as of {formatAsOfLabel(latestOddsAsOf)}</p> : null}
        {loading ? <p className="small">Loading games...</p> : null}
        {error ? <p className="small">Failed to load: {error}</p> : null}

        {!loading && !error && rows.length === 0 ? (
          <p className="small">No games scheduled today.</p>
        ) : null}

        {!loading && !error && rows.length > 0 ? (
          <table className={styles.gamesTable}>
            <thead>
              <tr>
                <th>Home Team</th>
                <th>Away Team</th>
                <th>Win Chance</th>
                <th>Moneyline</th>
                {league === "NBA" ? <th>Over 190</th> : null}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const side = expectedSide(row.home_win_probability);
                const chanceLabel = `${(expectedWinChance(row.home_win_probability, side) * 100).toFixed(1)}%`;
                return (
                  <tr key={row.game_id}>
                    <td className={side === "home" ? styles.teamWin : side === "away" ? styles.teamLoss : styles.teamNeutral}>
                      {row.home_team}
                    </td>
                    <td className={side === "away" ? styles.teamWin : side === "home" ? styles.teamLoss : styles.teamNeutral}>
                      {row.away_team}
                    </td>
                    <td className={styles.winChanceCell}>{side === "none" ? `Toss-up (${chanceLabel})` : chanceLabel}</td>
                    <td className={styles.moneylineCell}>
                      {`H ${formatMoneyline(row.home_moneyline)} · A ${formatMoneyline(row.away_moneyline)}`}
                    </td>
                    {league === "NBA" ? (
                      <td className={styles.over190Cell}>{formatOver190(row.over_190_price, row.over_190_point)}</td>
                    ) : null}
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : null}
      </div>
    </div>
  );
}

export default function GamesTodayPage() {
  return (
    <Suspense fallback={<p className="small">Loading games...</p>}>
      <GamesTodayPageContent />
    </Suspense>
  );
}
