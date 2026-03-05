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

function americanToImpliedProbability(odds: number): number | null {
  if (!Number.isFinite(odds) || odds === 0) return null;
  if (odds > 0) return 100 / (odds + 100);
  const absOdds = Math.abs(odds);
  return absOdds / (absOdds + 100);
}

function americanToDecimalOdds(odds: number): number | null {
  if (!Number.isFinite(odds) || odds === 0) return null;
  if (odds > 0) return 1 + odds / 100;
  return 1 + 100 / Math.abs(odds);
}

type BetDecision = {
  bet: string;
  reason: string;
};

function computeBetDecision(row: GamesTodayRow): BetDecision {
  const homeOdds = Number(row.home_moneyline);
  const awayOdds = Number(row.away_moneyline);
  if (!Number.isFinite(homeOdds) || !Number.isFinite(awayOdds) || homeOdds === 0 || awayOdds === 0) {
    return { bet: "$0", reason: "Missing odds" };
  }

  const pHomeRaw = Number(row.home_win_probability);
  if (!Number.isFinite(pHomeRaw)) return { bet: "$0", reason: "Missing odds" };
  const pHome = Math.min(1, Math.max(0, pHomeRaw));
  const pAway = 1 - pHome;
  if (Math.max(pHome, pAway) < 0.55) return { bet: "$0", reason: "Too close" };

  const impHome = americanToImpliedProbability(homeOdds);
  const impAway = americanToImpliedProbability(awayOdds);
  if (impHome === null || impAway === null) return { bet: "$0", reason: "Missing odds" };
  const impTotal = impHome + impAway;
  if (!Number.isFinite(impTotal) || impTotal <= 0) return { bet: "$0", reason: "Missing odds" };

  const fairHome = impHome / impTotal;
  const fairAway = impAway / impTotal;

  const decHome = americanToDecimalOdds(homeOdds);
  const decAway = americanToDecimalOdds(awayOdds);
  if (decHome === null || decAway === null) return { bet: "$0", reason: "Missing odds" };

  const evHome = pHome * decHome - 1;
  const evAway = pAway * decAway - 1;
  if (evHome <= 0 && evAway <= 0) return { bet: "$0", reason: "Price fair" };

  const side = evHome > evAway ? "home" : evAway > evHome ? "away" : pHome >= pAway ? "home" : "away";
  const modelProb = side === "home" ? pHome : pAway;
  const fairProb = side === "home" ? fairHome : fairAway;
  const edge = modelProb - fairProb;
  const ev = side === "home" ? evHome : evAway;
  if (edge < 0.03 || ev < 0.02) return { bet: "$0", reason: "Price fair" };

  const team = side === "home" ? row.home_team : row.away_team;
  const sideOdds = side === "home" ? homeOdds : awayOdds;
  const isUnderdog = sideOdds > 0;

  if (isUnderdog) {
    if (edge >= 0.08 && ev >= 0.10) return { bet: `$150 ${team}`, reason: "Underdog underpriced" };
    if (edge >= 0.05 && ev >= 0.05) return { bet: `$100 ${team}`, reason: "Underdog underpriced" };
    return { bet: `$50 ${team}`, reason: "Underdog underpriced" };
  }

  if (edge >= 0.08 && ev >= 0.10) return { bet: `$100 ${team}`, reason: "Favorite underpriced" };
  if (edge >= 0.05 && ev >= 0.05) return { bet: `$50 ${team}`, reason: "Favorite underpriced" };
  return { bet: "$0", reason: "Price fair" };
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
                {league === "NBA" ? <th>Over Odds</th> : null}
                <th>Bet per $100</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const side = expectedSide(row.home_win_probability);
                const chanceLabel = `${(expectedWinChance(row.home_win_probability, side) * 100).toFixed(1)}%`;
                const bet = computeBetDecision(row);
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
                    <td className={styles.betCell}>{bet.bet}</td>
                    <td className={styles.reasonCell}>{bet.reason}</td>
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
