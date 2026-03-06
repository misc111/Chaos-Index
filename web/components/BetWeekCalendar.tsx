"use client";

import type { HistoricalBetRow } from "@/lib/bet-history-types";
import { formatSignedUsd, formatUsd } from "@/lib/currency";
import styles from "./BetHistory.module.css";

type Props = {
  weekStart: string | null;
  bets: HistoricalBetRow[];
};

function parseDateKey(dateKey: string): Date {
  return new Date(`${dateKey}T12:00:00Z`);
}

function addDays(dateKey: string, days: number): string {
  const date = parseDateKey(dateKey);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

function formatDayLabel(dateKey: string): string {
  const parsed = parseDateKey(dateKey);
  if (Number.isNaN(parsed.getTime())) return dateKey;
  return parsed.toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatScore(homeScore: number | null, awayScore: number | null): string {
  if (!Number.isFinite(homeScore) || !Number.isFinite(awayScore)) return "Final score unavailable";
  return `${awayScore}-${homeScore} final`;
}

function amountClassName(value: number): string {
  return value > 0 ? styles.betAmountPositive : value < 0 ? styles.betAmountNegative : "";
}

function netClassName(value: number): string {
  if (value > 0) return `${styles.dayNet} ${styles.dayNetPositive}`;
  if (value < 0) return `${styles.dayNet} ${styles.dayNetNegative}`;
  return `${styles.dayNet} ${styles.dayNetNeutral}`;
}

export default function BetWeekCalendar({ weekStart, bets }: Props) {
  if (!weekStart) {
    return (
      <div className={`card ${styles.calendarCard}`}>
        <h2 className="title">Weekly Bet Calendar</h2>
        <p className={styles.emptyState}>A weekly replay calendar will appear once settled simulated bets are available.</p>
      </div>
    );
  }

  const days = Array.from({ length: 7 }, (_, index) => addDays(weekStart, index));
  const betsByDate = new Map<string, HistoricalBetRow[]>();
  for (const bet of bets) {
    const bucket = betsByDate.get(bet.date_central) || [];
    bucket.push(bet);
    betsByDate.set(bet.date_central, bucket);
  }

  return (
    <div className={styles.calendarGrid}>
      {days.map((dateKey) => {
        const dayBets = (betsByDate.get(dateKey) || []).slice().sort((left, right) => {
          const leftValue = left.start_time_utc || left.final_utc || "";
          const rightValue = right.start_time_utc || right.final_utc || "";
          return leftValue.localeCompare(rightValue) || left.game_id - right.game_id;
        });
        const dayProfit = dayBets.reduce((sum, bet) => sum + bet.profit, 0);
        const dayRisked = dayBets.reduce((sum, bet) => sum + bet.stake, 0);
        const dayClassName = dayBets.length ? styles.dayCard : `${styles.dayCard} ${styles.dayCardEmpty}`;

        return (
          <section key={dateKey} className={dayClassName}>
            <header className={styles.dayHeader}>
              <div className={styles.dayTitleRow}>
                <h3 className={styles.dayTitle}>{formatDayLabel(dateKey)}</h3>
                <span className={netClassName(dayProfit)}>{formatSignedUsd(dayProfit, { minimumFractionDigits: 2 })}</span>
              </div>
              <p className={styles.daySubtext}>
                {dayBets.length
                  ? `${dayBets.length} bet${dayBets.length === 1 ? "" : "s"} · Risked ${formatUsd(dayRisked, { minimumFractionDigits: 2 })}`
                  : "No simulated bets."}
              </p>
            </header>

            {dayBets.length ? (
              <div className={styles.betList}>
                {dayBets.map((bet) => {
                  const itemClassName =
                    bet.profit >= 0 ? `${styles.betItem} ${styles.betItemPositive}` : `${styles.betItem} ${styles.betItemNegative}`;
                  return (
                    <article key={bet.game_id} className={itemClassName}>
                      <div className={styles.betTop}>
                        <p className={styles.betMatchup}>{bet.away_team} at {bet.home_team}</p>
                        <p className={`${styles.betAmount} ${amountClassName(bet.profit)}`}>
                          {formatSignedUsd(bet.profit, { minimumFractionDigits: 2 })}
                        </p>
                      </div>
                      <p className={styles.betMeta}>
                        {bet.bet_label} at {bet.odds > 0 ? `+${Math.round(bet.odds)}` : Math.round(bet.odds)} · {formatScore(bet.home_score, bet.away_score)}
                      </p>
                      <p className={styles.betReason}>{bet.reason}</p>
                    </article>
                  );
                })}
              </div>
            ) : (
              <p className={styles.emptyState}>No bets landed on this day in the current replay window.</p>
            )}
          </section>
        );
      })}
    </div>
  );
}
