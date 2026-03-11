"use client";

import { Suspense, useMemo, useState } from "react";
import BetHistoryChart from "@/components/BetHistoryChart";
import styles from "@/components/BetHistory.module.css";
import BetWeekCalendar from "@/components/BetWeekCalendar";
import { BET_STRATEGIES, DEFAULT_BET_STRATEGY, getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";
import type { BetStrategyOptimizationSummary, ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import type { BetHistoryResponse, BetHistoryStrategyBundle } from "@/lib/bet-history-types";
import { HISTORICAL_BANKROLL_START_DATE_CENTRAL, HISTORICAL_BANKROLL_START_DOLLARS } from "@/lib/betting";
import { formatUsd } from "@/lib/currency";
import { useBetStrategy } from "@/lib/hooks/useBetStrategy";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";

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

function formatDateLabel(dateKey?: string | null): string {
  if (!dateKey) return "Replay start";
  const parsed = new Date(`${dateKey}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return dateKey;
  return parsed.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
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

const EMPTY_BET_HISTORY_STRATEGY: BetHistoryStrategyBundle = {
  summary: {
    total_final_games: 0,
    games_with_forecast: 0,
    games_with_odds: 0,
    analyzed_games: 0,
    suggested_bets: 0,
    wins: 0,
    losses: 0,
    total_risked: 0,
    total_profit: 0,
    roi: 0,
    starting_bankroll: HISTORICAL_BANKROLL_START_DOLLARS,
    current_bankroll: HISTORICAL_BANKROLL_START_DOLLARS,
    bankroll_start_central: HISTORICAL_BANKROLL_START_DATE_CENTRAL,
    coverage_start_central: null,
    coverage_end_central: null,
    note: "",
  },
  daily_points: [],
  bets: [],
};

function buildEmptyBetHistoryStrategies(): Record<BetStrategy, BetHistoryStrategyBundle> {
  return BET_STRATEGIES.reduce((acc, strategy) => {
    acc[strategy] = EMPTY_BET_HISTORY_STRATEGY;
    return acc;
  }, {} as Record<BetStrategy, BetHistoryStrategyBundle>);
}

function buildEmptyResolvedStrategyConfigs(): Record<BetStrategy, ResolvedBetStrategyConfig> {
  return BET_STRATEGIES.reduce((acc, strategy) => {
    acc[strategy] = {
      ...getBetStrategyConfig(strategy),
      config_signature: `empty|${strategy}`,
      optimization_objective: "Unavailable",
      optimization_source: "static_fallback",
      metrics: null,
    };
    return acc;
  }, {} as Record<BetStrategy, ResolvedBetStrategyConfig>);
}

function buildEmptyOptimizationSummary(): BetStrategyOptimizationSummary {
  return {
    method: "",
    risk_free_rate: 0,
    candidate_count: 0,
    frontier_point_count: 0,
    frontier: [],
    selected: BET_STRATEGIES.reduce((acc, strategy) => {
      acc[strategy] = null;
      return acc;
    }, {} as BetStrategyOptimizationSummary["selected"]),
  };
}

const EMPTY_BET_HISTORY: BetHistoryResponse = {
  league: "NHL",
  default_strategy: DEFAULT_BET_STRATEGY,
  strategy_configs: buildEmptyResolvedStrategyConfigs(),
  strategy_optimization: buildEmptyOptimizationSummary(),
  strategies: buildEmptyBetHistoryStrategies(),
};

function BetHistoryPageContent() {
  const league = useLeague();
  const strategy = useBetStrategy();
  const strategyConfig = getBetStrategyConfig(strategy);
  const { data, isLoading: loading, error } = useDashboardData<BetHistoryResponse>(
    "betHistory",
    "/api/bet-history",
    league,
    EMPTY_BET_HISTORY
  );
  const [selectedWeekStart, setSelectedWeekStart] = useState<string | null>(null);
  const activeHistory =
    data.strategies[strategy] || data.strategies[data.default_strategy] || EMPTY_BET_HISTORY.strategies.riskAdjusted;

  const weekStarts = useMemo(() => {
    return Array.from(new Set(activeHistory.bets.map((bet) => bet.week_start_central))).sort();
  }, [activeHistory]);
  const selectedWeek =
    selectedWeekStart && weekStarts.includes(selectedWeekStart) ? selectedWeekStart : (weekStarts[weekStarts.length - 1] ?? null);
  const selectedWeekIndex = selectedWeek ? weekStarts.indexOf(selectedWeek) : 0;
  const selectedWeekBets = useMemo(() => {
    if (!selectedWeek) return [];
    return activeHistory.bets.filter((bet) => bet.week_start_central === selectedWeek);
  }, [activeHistory, selectedWeek]);

  const summary = activeHistory.summary;
  const coverageLabel = formatDateRange(summary?.coverage_start_central, summary?.coverage_end_central);
  const bankrollStartLabel = formatDateLabel(summary?.bankroll_start_central);

  return (
    <div className="grid">
      <section className={`card ${styles.heroCard}`}>
        <div className={styles.heroTop}>
          <p className={styles.eyebrow}>Historical Replay</p>
          <h2 className="title">Bet History</h2>
          <p className={styles.heroText}>
            Replays settled games with stored pregame forecast and moneyline snapshots.
          </p>
          <p className={styles.heroText}>{strategyConfig.label} strategy.</p>
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
                <p className={styles.summarySubtext}>Across {summary.suggested_bets} settled bets under the current objective</p>
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
              <p className={styles.coverageText}>
                Bankroll path: {formatUsd(summary.starting_bankroll)} start on {bankrollStartLabel}
              </p>
              <p className={styles.coverageText}>Window: {coverageLabel}</p>
              <p className={styles.coverageText}>{summary.note}</p>
            </div>
          </>
        ) : null}
      </section>

      {!loading && !error ? <BetHistoryChart points={activeHistory.daily_points} strategyLabel={strategyConfig.label} /> : null}

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
              onClick={() => setSelectedWeekStart(weekStarts[Math.max(0, selectedWeekIndex - 1)] ?? null)}
              disabled={!weekStarts.length || selectedWeekIndex === 0}
            >
              Previous Week
            </button>
            <span className={styles.weekLabel}>{formatWeekLabel(selectedWeek)}</span>
            <button
              type="button"
              className={styles.weekButton}
              onClick={() => setSelectedWeekStart(weekStarts[Math.min(weekStarts.length - 1, selectedWeekIndex + 1)] ?? null)}
              disabled={!weekStarts.length || selectedWeekIndex >= weekStarts.length - 1}
            >
              Next Week
            </button>
          </div>
        </div>

        {loading ? <p className="small">Loading weekly replay calendar...</p> : null}
        {!loading && !error ? <BetWeekCalendar league={league} weekStart={selectedWeek} bets={selectedWeekBets} /> : null}
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
