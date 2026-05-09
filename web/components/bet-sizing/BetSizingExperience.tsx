"use client";

import { useMemo } from "react";
import { BetStakeWithIcon, TeamMatchup } from "@/components/TeamWithIcon";
import {
  computeBetDecisionsForSlate,
  formatBetRecommendation,
  REFERENCE_STAKE_DOLLARS,
  type BetInput,
  type BetDecision,
} from "@/lib/betting";
import { getBetStrategyConfig } from "@/lib/betting-strategy";
import { formatUsd } from "@/lib/currency";
import { centralTodayDateKey } from "@/lib/games-today";
import { resolveGamesTodayDateView } from "@/lib/games-today-view";
import { useBetStrategy } from "@/lib/hooks/useBetStrategy";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { LeagueCode } from "@/lib/league";
import type { GamesTodayResponse, GamesTodayRow } from "@/lib/types";
import styles from "./BetSizingExperience.module.css";

const EMPTY_GAMES_TODAY: GamesTodayResponse = {
  league: "MLB",
  as_of_utc: null,
  odds_as_of_utc: null,
  rows: [],
  historical_rows: [],
};

function formatAsOfLabel(value?: string | null): string {
  if (!value) return "Unknown";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatTipTime(value?: string | null): string {
  if (!value) return "Time TBD";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Chicago",
  });
}

function formatMoneyline(value?: number | null): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric === 0) return "No odds";
  const rounded = Math.round(numeric);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

function rowToDecisionInput(row: GamesTodayRow, league: LeagueCode): BetInput {
  return {
    league: league === "MLB" ? "MLB" : null,
    home_team: row.home_team,
    away_team: row.away_team,
    home_win_probability: row.home_win_probability,
    home_moneyline: row.home_moneyline,
    away_moneyline: row.away_moneyline,
    betting_model_name: row.betting_model_name,
    model_win_probabilities: row.model_win_probabilities,
  };
}

function replayDecisionFor(row: GamesTodayRow, strategy: string): BetDecision | null {
  const replay = row.replay_decisions?.[strategy as keyof NonNullable<GamesTodayRow["replay_decisions"]>];
  if (!replay) return null;
  return {
    bet: replay.bet_label,
    reason: replay.reason,
    side: replay.side,
    team: replay.team,
    stake: replay.stake,
    odds: replay.odds,
    modelProbability: replay.model_probability,
    marketProbability: replay.market_probability,
    edge: replay.edge,
    expectedValue: replay.expected_value,
  };
}

export default function BetSizingExperience() {
  const league = useLeague();
  const strategy = useBetStrategy(league);
  const gamesToday = useDashboardData<GamesTodayResponse>("gamesToday", "/api/games-today", league, EMPTY_GAMES_TODAY);
  const strategyConfig = gamesToday.data.strategy_configs?.[strategy] || getBetStrategyConfig(strategy, { league });
  const todayKey = centralTodayDateKey();
  const rows = useMemo(
    () =>
      resolveGamesTodayDateView({
        activeDateKey: gamesToday.data.date_central || todayKey,
        todayKey,
        upcomingRows: gamesToday.data.rows || [],
        historicalRows: gamesToday.data.historical_rows || [],
      }).rows,
    [gamesToday.data.date_central, gamesToday.data.historical_rows, gamesToday.data.rows, todayKey]
  );

  const decisions = useMemo(() => {
    const liveRows = rows.filter((row) => !replayDecisionFor(row, strategy));
    const liveDecisions = computeBetDecisionsForSlate(
      liveRows.map((row) => rowToDecisionInput(row, league)),
      strategy,
      strategyConfig
    );
    const liveByGame = new Map(liveRows.map((row, index) => [row.game_id, liveDecisions[index]]));
    return rows.map((row) => replayDecisionFor(row, strategy) || liveByGame.get(row.game_id) || null);
  }, [league, rows, strategy, strategyConfig]);

  const betRows = rows
    .map((row, index) => ({ row, decision: decisions[index] }))
    .filter(({ decision }) => Boolean(decision && decision.stake > 0 && decision.side !== "none"));
  const noBetCount = Math.max(0, rows.length - betRows.length);
  const totalAtRisk = betRows.length * REFERENCE_STAKE_DOLLARS;

  return (
    <div className="grid">
      <section className={`card ${styles.heroCard}`}>
        <div className={styles.heroTop}>
          <div>
            <p className={styles.eyebrow}>Flat Betting Rule</p>
            <h2 className="title">Bet {formatUsd(REFERENCE_STAKE_DOLLARS)} or bet nothing.</h2>
            <p className={styles.heroText}>
              The model still checks the price first. Once a game passes that screen, the stake is always the same fixed amount.
              If the game does not pass, the stake is $0.
            </p>
          </div>
          <div className={styles.heroMeta}>
            <span className={styles.heroPill}>{league}</span>
            <span className={styles.heroPill}>{strategyConfig.label}</span>
          </div>
        </div>

        <div className={styles.heroMetrics}>
          <div className={styles.heroMetric}>
            <span className={styles.heroMetricLabel}>Amount per bet</span>
            <strong className={styles.heroMetricValue}>{formatUsd(REFERENCE_STAKE_DOLLARS)}</strong>
          </div>
          <div className={styles.heroMetric}>
            <span className={styles.heroMetricLabel}>Bets today</span>
            <strong className={styles.heroMetricValue}>{betRows.length}</strong>
          </div>
          <div className={styles.heroMetric}>
            <span className={styles.heroMetricLabel}>No-bet games</span>
            <strong className={styles.heroMetricValue}>{noBetCount}</strong>
          </div>
          <div className={styles.heroMetric}>
            <span className={styles.heroMetricLabel}>Total at risk</span>
            <strong className={styles.heroMetricValue}>{formatUsd(totalAtRisk)}</strong>
          </div>
        </div>

        <div className={styles.stageGrid}>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>1. Check the price</p>
            <p className={styles.stageTitle}>The model needs usable sportsbook odds.</p>
            <p className={styles.stageBody}>If the page has no odds for a game, it is automatically a no-bet.</p>
          </article>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>2. Decide bet or no bet</p>
            <p className={styles.stageTitle}>The model compares its estimate to the sportsbook price.</p>
            <p className={styles.stageBody}>A bet only appears when the model sees enough advantage after its uncertainty checks.</p>
          </article>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>3. Use the fixed stake</p>
            <p className={styles.stageTitle}>Every accepted bet gets the same amount.</p>
            <p className={styles.stageBody}>There is no bigger-or-smaller sizing layer. A bet is {formatUsd(REFERENCE_STAKE_DOLLARS)}; a pass is $0.</p>
          </article>
        </div>

        <div className={styles.asOfRow}>
          <span className="small">Forecast snapshot as of {formatAsOfLabel(gamesToday.data.as_of_utc)}</span>
          <span className="small">Odds snapshot as of {formatAsOfLabel(gamesToday.data.odds_as_of_utc)}</span>
        </div>
      </section>

      {gamesToday.isLoading ? <p className="small">Loading flat bet view...</p> : null}
      {gamesToday.error ? <p className="small">Failed to load flat bet data: {gamesToday.error}</p> : null}

      {!gamesToday.isLoading && !gamesToday.error ? (
        <section className={`card ${styles.gamesCard}`}>
          <div className={styles.gamesHeader}>
            <div>
              <p className={styles.eyebrow}>Today</p>
              <h2 className="title">Bet List</h2>
              <p className="small">These are the games currently receiving the fixed stake.</p>
            </div>
            <div className={styles.gamesPills}>
              <span className={styles.heroPill}>{betRows.length} bets</span>
              <span className={styles.heroPill}>{noBetCount} no-bets</span>
            </div>
          </div>

          <div className={styles.gameList}>
            {betRows.length ? (
              betRows.map(({ row, decision }) => {
                const display = formatBetRecommendation({
                  team: decision?.team || null,
                  stake: decision?.stake || 0,
                  reason: decision?.reason || "",
                });
                return (
                  <article key={row.game_id} className={styles.gameButton}>
                    <div className={styles.gameButtonTop}>
                      <span className={styles.tipBadge}>{formatTipTime(row.start_time_utc)} CT</span>
                      <BetStakeWithIcon
                        league={league}
                        teamCode={decision?.team}
                        label={decision?.team}
                        stake={decision?.stake || 0}
                        zeroLabel="No bet"
                      />
                    </div>
                    <TeamMatchup
                      league={league}
                      awayTeamCode={row.away_team}
                      homeTeamCode={row.home_team}
                      awayLabel={row.away_team}
                      homeLabel={row.home_team}
                      size="sm"
                    />
                    <div className={styles.gameStats}>
                      <span>Home {formatMoneyline(row.home_moneyline)}</span>
                      <span>Away {formatMoneyline(row.away_moneyline)}</span>
                      <span>{display.label}</span>
                    </div>
                    <p className={styles.gameSummary}>{display.reason}</p>
                  </article>
                );
              })
            ) : (
              <article className={styles.stageCard}>
                <p className={styles.stageTitle}>No fixed-stake bets right now.</p>
                <p className={styles.stageBody}>The current slate is all passes, or odds are not available yet.</p>
              </article>
            )}
          </div>
        </section>
      ) : null}
    </div>
  );
}
