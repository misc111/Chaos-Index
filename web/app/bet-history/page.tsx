"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import BetHistoryChart from "@/components/BetHistoryChart";
import styles from "@/components/BetHistory.module.css";
import BetWeekCalendar from "@/components/BetWeekCalendar";
import type { BetHistoryResponse } from "@/lib/bet-history-types";
import { formatUsd } from "@/lib/currency";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDateRange(startDate?: string | null, endDate?: string | null): string {
  if (!startDate || !endDate) return "Coverage window opens once replayable games exist.";

  const start = new Date(`${startDate}T12:00:00Z`);
  const end = new Date(`${endDate}T12:00:00Z`);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    return `${startDate} through ${endDate}`;
  }

  const formatter = new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return `${formatter.format(start)} through ${formatter.format(end)}`;
}

function formatWeekLabel(weekStart: string | null): string {
  if (!weekStart) return "No replay week";
  const start = new Date(`${weekStart}T12:00:00Z`);
  if (Number.isNaN(start.getTime())) return weekStart;
  const end = new Date(start);
  end.setUTCDate(end.getUTCDate() + 6);
  return `${start.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  })} - ${end.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  })}`;
}

function valueClassName(value: number): string {
  if (value > 0) return `${styles.summaryValue} ${styles.summaryValuePositive}`;
  if (value < 0) return `${styles.summaryValue} ${styles.summaryValueNegative}`;
  return styles.summaryValue;
}

function BetHistoryPageContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const [data, setData] = useState<BetHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedWeekIndex, setSelectedWeekIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetchDashboardJson<BetHistoryResponse>("betHistory", "/api/bet-history", league)
      .then((payload) => {
        if (cancelled) return;
        setData(payload);
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

  const weekStarts = useMemo(() => {
    if (!data) return [] as string[];
    return Array.from(new Set(data.bets.map((bet) => bet.week_start_central))).sort();
  }, [data]);

  useEffect(() => {
    if (!weekStarts.length) {
      setSelectedWeekIndex(0);
      return;
    }
    setSelectedWeekIndex((current) => {
      if (current >= 0 && current < weekStarts.length) return current;
      return weekStarts.length - 1;
    });
  }, [weekStarts]);

  const selectedWeek = weekStarts[selectedWeekIndex] || null;
  const selectedWeekBets = useMemo(() => {
    if (!data || !selectedWeek) return [];
    return data.bets.filter((bet) => bet.week_start_central === selectedWeek);
  }, [data, selectedWeek]);

  const summary = data?.summary;
  const coverageLabel = formatDateRange(summary?.coverage_start_central, summary?.coverage_end_central);

  return (
    <div className="grid">
      <section className={`card ${styles.heroCard}`}>
        <div className={styles.heroTop}>
          <p className={styles.eyebrow}>Historical Replay</p>
          <h2 className="title">Bet History</h2>
          <p className={styles.heroText}>
            This screen replays the Games Today bet logic against finalized games whenever the database contains both a pregame forecast snapshot and a matching pregame moneyline snapshot.
          </p>
        </div>

        {loading ? <p className="small">Loading replay history...</p> : null}
        {error ? <p className="small">Failed to load replay history: {error}</p> : null}

        {!loading && !error && summary ? (
          <>
            <div className={styles.summaryGrid}>
              <article className={styles.summaryTile}>
                <p className={styles.summaryLabel}>Net P/L</p>
                <p className={valueClassName(summary.total_profit)}>
                  {formatUsd(summary.total_profit, { minimumFractionDigits: 2 })}
                </p>
                <p className={styles.summarySubtext}>Across {summary.suggested_bets} settled bets</p>
              </article>

              <article className={styles.summaryTile}>
                <p className={styles.summaryLabel}>ROI</p>
                <p className={styles.summaryValue}>{formatPercent(summary.roi)}</p>
                <p className={styles.summarySubtext}>
                  Risked {formatUsd(summary.total_risked, { minimumFractionDigits: 2 })}
                </p>
              </article>

              <article className={styles.summaryTile}>
                <p className={styles.summaryLabel}>Record</p>
                <p className={styles.summaryValue}>{summary.wins}-{summary.losses}</p>
                <p className={styles.summarySubtext}>Wins vs. losses on simulated bets</p>
              </article>

              <article className={styles.summaryTile}>
                <p className={styles.summaryLabel}>Coverage</p>
                <p className={styles.summaryValue}>{summary.analyzed_games}/{summary.total_final_games}</p>
                <p className={styles.summarySubtext}>Final games with replayable pregame inputs</p>
              </article>
            </div>

            <div className={styles.coverageBar}>
              <p className={styles.coverageText}>Window: {coverageLabel}</p>
              <p className={styles.coverageText}>{summary.note}</p>
            </div>
          </>
        ) : null}
      </section>

      {!loading && !error && data ? <BetHistoryChart points={data.daily_points} /> : null}

      <section className={`card ${styles.calendarCard}`}>
        <div className={styles.calendarHeader}>
          <div>
            <h2 className="title">Weekly Bet Calendar</h2>
            <p className="small">Tall day cards show each simulated wager and the settled profit or loss per game.</p>
          </div>

          <div className={styles.weekControls}>
            <button
              type="button"
              className={styles.weekButton}
              onClick={() => setSelectedWeekIndex((current) => Math.max(0, current - 1))}
              disabled={!weekStarts.length || selectedWeekIndex === 0}
            >
              Previous Week
            </button>
            <span className={styles.weekLabel}>{formatWeekLabel(selectedWeek)}</span>
            <button
              type="button"
              className={styles.weekButton}
              onClick={() => setSelectedWeekIndex((current) => Math.min(weekStarts.length - 1, current + 1))}
              disabled={!weekStarts.length || selectedWeekIndex >= weekStarts.length - 1}
            >
              Next Week
            </button>
          </div>
        </div>

        {loading ? <p className="small">Loading weekly replay calendar...</p> : null}
        {!loading && !error ? <BetWeekCalendar weekStart={selectedWeek} bets={selectedWeekBets} /> : null}
      </section>
    </div>
  );
}

export default function BetHistoryPage() {
  return (
    <Suspense fallback={<p className="small">Loading bet history...</p>}>
      <BetHistoryPageContent />
    </Suspense>
  );
}
