"use client";

import TeamWithIcon from "@/components/TeamWithIcon";
import type { HistoricalBetRow } from "@/lib/bet-history-types";
import { formatSignedUsd, formatUsd } from "@/lib/currency";
import type { LeagueCode } from "@/lib/league";
import styles from "./BetHistory.module.css";

type Props = {
  league: LeagueCode;
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
    month: "numeric",
    day: "numeric",
    year: "numeric",
  });
}

function formatRisked(value: number): string {
  return formatUsd(value, {
    minimumFractionDigits: Number.isInteger(value) ? 0 : 2,
    maximumFractionDigits: 2,
  });
}

function formatTeamCode(value: string | null | undefined): string {
  const teamCode = String(value || "").trim();
  return teamCode || "TBD";
}

function amountClassName(value: number): string {
  return value > 0 ? styles.amountPositive : value < 0 ? styles.amountNegative : "";
}

function netClassName(value: number): string {
  if (value > 0) return `${styles.dayNet} ${styles.dayNetPositive}`;
  if (value < 0) return `${styles.dayNet} ${styles.dayNetNegative}`;
  return `${styles.dayNet} ${styles.dayNetNeutral}`;
}

export default function BetWeekCalendar({ league, weekStart, bets }: Props) {
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
              <h3 className={styles.dayTitle}>{formatDayLabel(dateKey)}</h3>
              <p className={netClassName(dayProfit)}>{formatSignedUsd(dayProfit, { minimumFractionDigits: 2 })}</p>
              {dayBets.length ? (
                <p className={styles.riskBadge}>Risked {formatRisked(dayRisked)}</p>
              ) : (
                <p className={styles.emptyState}>No bets.</p>
              )}
            </header>

            {dayBets.length ? (
              <div className={styles.betList}>
                {dayBets.map((bet) => {
                  const itemClassName =
                    bet.profit >= 0 ? `${styles.betItem} ${styles.betItemPositive}` : `${styles.betItem} ${styles.betItemNegative}`;
                  return (
                    <article key={bet.game_id} className={itemClassName}>
                      <p className={`${styles.amountValue} ${styles.betAmount} ${amountClassName(bet.profit)}`}>
                        {formatSignedUsd(bet.profit, { minimumFractionDigits: 2 })}
                      </p>
                      <p className={styles.betRisk}>Risked {formatRisked(bet.stake)}</p>
                      <div className={styles.betTeams} aria-label={`${formatTeamCode(bet.away_team)} at ${formatTeamCode(bet.home_team)}`}>
                        <TeamWithIcon
                          league={league}
                          teamCode={bet.away_team}
                          label={formatTeamCode(bet.away_team)}
                          className={styles.betTeamRow}
                          textClassName={styles.betTeamText}
                        />
                        <span className={styles.betTeamsSeparator}>@</span>
                        <TeamWithIcon
                          league={league}
                          teamCode={bet.home_team}
                          label={formatTeamCode(bet.home_team)}
                          className={styles.betTeamRow}
                          textClassName={styles.betTeamText}
                        />
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
