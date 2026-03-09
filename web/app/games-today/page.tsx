"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  getBetStrategyConfig,
  normalizeBetStrategy,
  type BetStrategy,
} from "@/lib/betting-strategy";
import {
  REFERENCE_STAKE_DOLLARS,
  REFERENCE_BANKROLL_DOLLARS,
  computeBetDecisionsForSlate,
  expectedSide,
  expectedWinChance,
  formatBetRecommendation,
} from "@/lib/betting";
import type { ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import TeamWithIcon, { BetStakeWithIcon, TeamMatchup } from "@/components/TeamWithIcon";
import {
  centralTodayDateKey,
  dateKeyForScheduledGame,
  formatCentralDateLabel,
  formatCentralDateSummary,
  normalizeCentralDateKey,
  normalizeUtcTimestamp,
  shiftCentralDateKey,
} from "@/lib/games-today";
import { normalizeLeague, withLeague } from "@/lib/league";
import { fetchDashboardJson, isStaticStagingBuild } from "@/lib/static-staging";
import type { GamesTodayResponse, GamesTodayRow } from "@/lib/types";
import { formatUsd } from "@/lib/currency";
import styles from "./styles.module.css";

type BetRecommendationDisplay = {
  label: string;
  reason: string;
  team: string | null;
  stake: number;
};

type RefreshOddsResponse = {
  ok?: boolean;
  error?: string;
  details?: string;
  odds_as_of_utc?: string | null;
  event_count?: number | null;
  row_count?: number | null;
};

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

function formatCentralTip(value?: string | null): string {
  if (!value) return "Time TBD";
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Chicago",
    timeZoneName: "short",
  });
}

function displayBetRecommendation(
  row: GamesTodayRow,
  liveDecisionMap: Map<number, BetRecommendationDisplay>,
  strategy: BetStrategy
): BetRecommendationDisplay {
  const replayDecision = row.replay_decisions?.[strategy];
  if (replayDecision) {
    return {
      ...formatBetRecommendation(replayDecision),
      team: replayDecision.team,
      stake: replayDecision.stake,
    };
  }
  return liveDecisionMap.get(row.game_id) || {
    label: "$0",
    reason: "Missing odds",
    team: null,
    stake: 0,
  };
}

function latestTimestamp(rows: GamesTodayRow[], field: "forecast_as_of_utc" | "odds_as_of_utc"): string {
  let latestValue = "";
  let latestMillis = Number.NEGATIVE_INFINITY;

  for (const row of rows) {
    const value = String(row[field] || "").trim();
    if (!value) continue;
    const parsed = new Date(value);
    const millis = parsed.getTime();
    if (Number.isNaN(millis) || millis <= latestMillis) continue;
    latestMillis = millis;
    latestValue = value;
  }

  return latestValue;
}

function GamesTodayPageContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const strategy = normalizeBetStrategy(searchParams.get("strategy"));
  const staticStaging = isStaticStagingBuild();
  const [upcomingRows, setUpcomingRows] = useState<GamesTodayRow[]>([]);
  const [historicalRows, setHistoricalRows] = useState<GamesTodayRow[]>([]);
  const [strategyConfigs, setStrategyConfigs] = useState<Record<BetStrategy, ResolvedBetStrategyConfig> | undefined>(undefined);
  const [latestAsOf, setLatestAsOf] = useState<string>("");
  const [historicalCoverageStart, setHistoricalCoverageStart] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [refreshingOdds, setRefreshingOdds] = useState(false);
  const [refreshOddsError, setRefreshOddsError] = useState("");
  const [refreshOddsStatus, setRefreshOddsStatus] = useState("");
  const [selectedDateKey, setSelectedDateKey] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetchDashboardJson<GamesTodayResponse>("gamesToday", "/api/games-today", league)
      .then((payload) => {
        if (cancelled) return;
        setUpcomingRows(payload.rows || []);
        setHistoricalRows(payload.historical_rows || []);
        setStrategyConfigs(payload.strategy_configs);
        setLatestAsOf(typeof payload.as_of_utc === "string" ? payload.as_of_utc : "");
        setHistoricalCoverageStart(
          typeof payload.historical_coverage_start_central === "string" ? payload.historical_coverage_start_central : ""
        );
        setSelectedDateKey((current) => {
          if (current) return current;
          return normalizeCentralDateKey(payload.date_central) || centralTodayDateKey();
        });
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
  }, [league, reloadKey]);

  const todayKey = centralTodayDateKey();
  const activeDateKey = selectedDateKey || todayKey;
  const isPastDate = activeDateKey < todayKey;
  // Maintainer note: the page keeps one table/card layout across leagues.
  // We swap the backing dataset by date instead of rendering separate NHL/NBA
  // or past/future views. Add shared fields to both payload pools if needed.
  const sourceRows = isPastDate ? historicalRows : upcomingRows;
  const rows = sourceRows.filter((row) => dateKeyForScheduledGame(row) === activeDateKey);
  const selectedForecastAsOf = latestTimestamp(rows, "forecast_as_of_utc") || latestAsOf;
  const selectedOddsAsOf = latestTimestamp(rows, "odds_as_of_utc");
  const liveDecisionMap = useMemo(() => {
    const liveRows = rows.filter((row) => !row.replay_decisions?.[strategy]);
    if (!liveRows.length) return new Map<number, BetRecommendationDisplay>();

    const resolvedConfig = strategyConfigs?.[strategy] || getBetStrategyConfig(strategy);
    const decisions = computeBetDecisionsForSlate(liveRows, strategy, resolvedConfig);
    return new Map(
      liveRows.map((row, index) => [
        row.game_id,
        {
          ...formatBetRecommendation(decisions[index]),
          team: decisions[index]?.team ?? null,
          stake: decisions[index]?.stake ?? 0,
        },
      ])
    );
  }, [rows, strategy, strategyConfigs]);
  const scheduleSummary = formatCentralDateSummary(activeDateKey);
  const dateLabel = formatCentralDateLabel(activeDateKey);
  const title = activeDateKey === todayKey ? "Games Today" : `Games on ${dateLabel}`;
  const description = isPastDate
    ? `Stored pregame replay rows for ${dateLabel} (Central Time) are shown.`
    : `Only games scheduled for ${scheduleSummary} (Central Time) are shown.`;
  const emptyState =
    isPastDate && historicalCoverageStart && activeDateKey < historicalCoverageStart
      ? `No stored pregame replay data for ${dateLabel}. Replay coverage starts on ${formatCentralDateLabel(historicalCoverageStart)}.`
      : isPastDate
        ? `No replayable games available for ${dateLabel}.`
        : `No games scheduled for ${scheduleSummary}.`;

  const handleRefreshOdds = async () => {
    if (staticStaging) {
      setRefreshOddsStatus("Odds refresh is disabled in the GitHub Pages staging build.");
      setRefreshOddsError("");
      return;
    }

    setRefreshingOdds(true);
    setRefreshOddsError("");
    setRefreshOddsStatus("");

    try {
      const response = await fetch(withLeague("/api/refresh-odds", league), {
        method: "POST",
      });
      const payload = (await response.json().catch(() => ({}))) as RefreshOddsResponse;
      if (!response.ok || payload.ok === false) {
        const detailSuffix = payload.details ? ` ${payload.details}` : "";
        throw new Error(payload.error ? `${payload.error}${detailSuffix}` : `Odds refresh failed (${response.status}).`);
      }

      if (typeof payload.odds_as_of_utc === "string" && payload.odds_as_of_utc) {
        const counts =
          payload.event_count != null && payload.row_count != null
            ? ` Saved ${Number(payload.event_count)} events and ${Number(payload.row_count)} lines.`
            : "";
        setRefreshOddsStatus(`Odds refreshed as of ${formatAsOfLabel(payload.odds_as_of_utc)}.${counts}`);
      } else {
        setRefreshOddsStatus("Odds refreshed and saved.");
      }

      setReloadKey((value) => value + 1);
    } catch (refreshError) {
      const message = refreshError instanceof Error ? refreshError.message : "Unable to refresh odds right now.";
      setRefreshOddsError(message.length > 320 ? `${message.slice(0, 317)}...` : message);
    } finally {
      setRefreshingOdds(false);
    }
  };

  return (
    <div className="grid">
      <div className={`card ${styles.tableCard}`}>
        <h2 className="title">{title}</h2>
        <div className={styles.dayNavRow}>
          <button
            type="button"
            className={styles.dayNavButton}
            onClick={() => setSelectedDateKey((current) => shiftCentralDateKey(current || centralTodayDateKey(), -1))}
            disabled={loading}
            aria-label="Show the previous day"
          >
            Previous
          </button>
          <button
            type="button"
            className={styles.dayNavButton}
            onClick={() => setSelectedDateKey((current) => shiftCentralDateKey(current || centralTodayDateKey(), 1))}
            disabled={loading}
            aria-label="Show the next day"
          >
            Next
          </button>
        </div>
        <p className="small">{description}</p>
        <p className="small">
          Stakes use uncertainty-adjusted edge and a bankroll-linked scale. A {formatUsd(REFERENCE_STAKE_DOLLARS)} recommendation corresponds to 1% of the ${REFERENCE_BANKROLL_DOLLARS.toLocaleString()} reference bankroll.
        </p>
        <div className={styles.actionsRow}>
          <button
            type="button"
            className={styles.refreshOddsButton}
            onClick={handleRefreshOdds}
            disabled={refreshingOdds || staticStaging}
            aria-busy={refreshingOdds}
          >
            {staticStaging ? "Snapshot Only" : refreshingOdds ? "Refreshing odds..." : "Refresh Odds"}
          </button>
          {refreshOddsStatus ? <p className={styles.refreshStatus}>{refreshOddsStatus}</p> : null}
          {refreshOddsError ? <p className={styles.refreshError}>{refreshOddsError}</p> : null}
        </div>
        {selectedForecastAsOf ? <p className="small">Forecast snapshot as of {formatAsOfLabel(selectedForecastAsOf)}</p> : null}
        {selectedOddsAsOf ? <p className="small">Odds snapshot as of {formatAsOfLabel(selectedOddsAsOf)}</p> : null}
        {loading ? <p className="small">Loading games...</p> : null}
        {error ? <p className="small">Failed to load: {error}</p> : null}

        {!loading && !error && rows.length === 0 ? (
          <p className="small">{emptyState}</p>
        ) : null}

        {!loading && !error && rows.length > 0 ? (
          <>
            <div className={styles.tableDesktop}>
              <table className={styles.gamesTable}>
                <thead>
                  <tr>
                    <th>Home Team</th>
                    <th>Away Team</th>
                    <th>Time (CST/CDT)</th>
                    <th>Win Chance</th>
                    <th>Moneyline</th>
                    {/* Maintainer note: this is the one league-specific column in the shared table. */}
                    {league === "NBA" ? <th>Over Odds</th> : null}
                    <th>Suggested Bet</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => {
                    const side = expectedSide(row.home_win_probability);
                    const chanceLabel = `${(expectedWinChance(row.home_win_probability, side) * 100).toFixed(1)}%`;
                    const bet = displayBetRecommendation(row, liveDecisionMap, strategy);
                    return (
                      <tr key={row.game_id}>
                        <td className={side === "home" ? styles.teamWin : side === "away" ? styles.teamLoss : styles.teamNeutral}>
                          <TeamWithIcon league={league} teamCode={row.home_team} label={row.home_team} />
                        </td>
                        <td className={side === "away" ? styles.teamWin : side === "home" ? styles.teamLoss : styles.teamNeutral}>
                          <TeamWithIcon league={league} teamCode={row.away_team} label={row.away_team} />
                        </td>
                        <td className={styles.timeCell}>{formatCentralTip(row.start_time_utc)}</td>
                        <td className={styles.winChanceCell}>{chanceLabel}</td>
                        <td className={styles.moneylineCell}>
                          {`H ${formatMoneyline(row.home_moneyline)} · A ${formatMoneyline(row.away_moneyline)}`}
                        </td>
                        {league === "NBA" ? (
                          <td className={styles.over190Cell}>{formatOver190(row.over_190_price, row.over_190_point)}</td>
                        ) : null}
                        <td className={styles.betCell}>
                          <BetStakeWithIcon league={league} teamCode={bet.team} label={bet.team} stake={bet.stake} />
                        </td>
                        <td className={styles.reasonCell}>{bet.reason}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className={styles.tableMobile}>
              <div className={styles.mobileCardList}>
                {rows.map((row) => {
                  const side = expectedSide(row.home_win_probability);
                  const chanceLabel = `${(expectedWinChance(row.home_win_probability, side) * 100).toFixed(1)}%`;
                  const bet = displayBetRecommendation(row, liveDecisionMap, strategy);
                  return (
                    <article key={`${row.game_id}-mobile`} className={styles.mobileCard}>
                      <div className={styles.mobileCardTop}>
                        <div>
                          <p className={styles.mobileCardEyebrow}>Matchup</p>
                          <h3 className={styles.mobileCardTitle}>
                            <TeamMatchup
                              league={league}
                              awayTeamCode={row.away_team}
                              homeTeamCode={row.home_team}
                              awayLabel={row.away_team}
                              homeLabel={row.home_team}
                              size="md"
                            />
                          </h3>
                        </div>
                        <span
                          className={`${styles.mobileSideBadge} ${
                            side === "home" ? styles.teamWin : side === "away" ? styles.teamLoss : styles.teamNeutral
                          }`}
                        >
                          {side === "home" ? (
                            <span className={styles.mobileSideContent}>
                              <TeamWithIcon league={league} teamCode={row.home_team} label={row.home_team} />
                              <span>lean</span>
                            </span>
                          ) : side === "away" ? (
                            <span className={styles.mobileSideContent}>
                              <TeamWithIcon league={league} teamCode={row.away_team} label={row.away_team} />
                              <span>lean</span>
                            </span>
                          ) : (
                            "Even"
                          )}
                        </span>
                      </div>

                      <div className={styles.mobileMetaGrid}>
                        <div className={styles.mobileMetaItem}>
                          <span className={styles.mobileMetaLabel}>Tip (CST/CDT)</span>
                          <span className={styles.mobileMetaValue}>{formatCentralTip(row.start_time_utc)}</span>
                        </div>
                        <div className={styles.mobileMetaItem}>
                          <span className={styles.mobileMetaLabel}>Win chance</span>
                          <span className={styles.mobileMetaValue}>{chanceLabel}</span>
                        </div>
                        <div className={styles.mobileMetaItem}>
                          <span className={styles.mobileMetaLabel}>Moneyline</span>
                          <span className={styles.mobileMetaValue}>
                            {`H ${formatMoneyline(row.home_moneyline)} · A ${formatMoneyline(row.away_moneyline)}`}
                          </span>
                        </div>
                        {league === "NBA" ? (
                          <div className={styles.mobileMetaItem}>
                            <span className={styles.mobileMetaLabel}>Over odds</span>
                            <span className={styles.mobileMetaValue}>
                              {formatOver190(row.over_190_price, row.over_190_point)}
                            </span>
                          </div>
                        ) : null}
                        <div className={styles.mobileMetaItem}>
                          <span className={styles.mobileMetaLabel}>Suggested Bet</span>
                          <span className={styles.mobileMetaValue}>
                            <BetStakeWithIcon league={league} teamCode={bet.team} label={bet.team} stake={bet.stake} />
                          </span>
                        </div>
                        <div className={styles.mobileMetaItem}>
                          <span className={styles.mobileMetaLabel}>Reason</span>
                          <span className={styles.mobileMetaValue}>{bet.reason}</span>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>
          </>
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
