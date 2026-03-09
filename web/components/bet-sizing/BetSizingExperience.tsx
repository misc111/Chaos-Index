"use client";

import { useMemo, useState } from "react";
import BetSizingFrontier from "@/components/BetSizingFrontier";
import BetSizingBudgetFlow from "@/components/bet-sizing/BetSizingBudgetFlow";
import BetSizingProbabilityBridge from "@/components/bet-sizing/BetSizingProbabilityBridge";
import BetSizingStakeFlow from "@/components/bet-sizing/BetSizingStakeFlow";
import { BetStakeWithIcon, TeamMatchup } from "@/components/TeamWithIcon";
import { buildBetSizingExplainerModel } from "@/lib/bet-sizing-explainer";
import {
  buildBetSizingGamePreviews,
  collectBetSizingPolicies,
  selectBetSizingSlate,
  selectDefaultGameId,
  selectDefaultPolicyKey,
} from "@/lib/bet-sizing-view";
import {
  BET_UNIT_BANKROLL_FRACTION,
  BET_UNIT_DOLLARS,
  HISTORICAL_BANKROLL_START_DATE_CENTRAL,
  HISTORICAL_BANKROLL_START_DOLLARS,
  REFERENCE_BANKROLL_DOLLARS,
} from "@/lib/betting";
import type { BetStrategyPerformanceSnapshot, FrontierPointSummary, ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import type { BetHistoryResponse, BetHistoryStrategyBundle } from "@/lib/bet-history-types";
import { BET_STRATEGIES, getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";
import { formatUsd } from "@/lib/currency";
import { useBetStrategy } from "@/lib/hooks/useBetStrategy";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { GamesTodayResponse } from "@/lib/types";
import styles from "./BetSizingExperience.module.css";

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatUnits(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)}u`;
}

function formatBankrollShare(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * BET_UNIT_BANKROLL_FRACTION * 100).toFixed(2)}%`;
}

function formatProbabilityPoints(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pts`;
}

function formatExpectedValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatMoneyline(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return "—";
  const rounded = Math.round(value);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

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

function metricOrDash(
  metrics: BetStrategyPerformanceSnapshot | FrontierPointSummary | null | undefined,
  accessor: (metrics: BetStrategyPerformanceSnapshot | FrontierPointSummary) => number,
  formatter: (value: number) => string
): string {
  if (!metrics) return "—";
  const value = accessor(metrics);
  return Number.isFinite(value) ? formatter(value) : "—";
}

function sourceLabel(value: string): string {
  switch (value) {
    case "historical_frontier":
      return "Replay-ranked";
    case "historical_downside":
      return "Downside-ranked";
    case "static_fallback":
      return "Static default";
    default:
      return "Replay preview";
  }
}

function screeningTone(current: number, previous: number): string {
  if (current === previous) return styles.screenStepStable;
  if (current === 0) return styles.screenStepEmpty;
  return styles.screenStepActive;
}

function buildEmptyStrategyConfigs(): Record<BetStrategy, ResolvedBetStrategyConfig> {
  return {
    riskAdjusted: {
      ...getBetStrategyConfig("riskAdjusted"),
      config_signature: "riskAdjusted",
      optimization_objective: "Unavailable",
      optimization_source: "static_fallback",
      metrics: null,
    },
    aggressive: {
      ...getBetStrategyConfig("aggressive"),
      config_signature: "aggressive",
      optimization_objective: "Unavailable",
      optimization_source: "static_fallback",
      metrics: null,
    },
    capitalPreservation: {
      ...getBetStrategyConfig("capitalPreservation"),
      config_signature: "capitalPreservation",
      optimization_objective: "Unavailable",
      optimization_source: "static_fallback",
      metrics: null,
    },
  };
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

const EMPTY_BET_HISTORY: BetHistoryResponse = {
  league: "NHL",
  default_strategy: "riskAdjusted",
  strategy_configs: buildEmptyStrategyConfigs(),
  strategy_optimization: {
    method: "",
    risk_free_rate: 0,
    candidate_count: 0,
    frontier_point_count: 0,
    frontier: [],
    selected: {
      riskAdjusted: null,
      aggressive: null,
      capitalPreservation: null,
    },
  },
  strategies: {
    riskAdjusted: EMPTY_BET_HISTORY_STRATEGY,
    aggressive: EMPTY_BET_HISTORY_STRATEGY,
    capitalPreservation: EMPTY_BET_HISTORY_STRATEGY,
  },
};

const EMPTY_GAMES_TODAY: GamesTodayResponse = {
  league: "NHL",
  as_of_utc: null,
  odds_as_of_utc: null,
  date_central: undefined,
  historical_coverage_start_central: null,
  strategy_configs: buildEmptyStrategyConfigs(),
  strategy_optimization: EMPTY_BET_HISTORY.strategy_optimization,
  historical_rows: [],
  rows: [],
};

export default function BetSizingExperience() {
  const league = useLeague();
  const strategy = useBetStrategy();
  const betHistory = useDashboardData<BetHistoryResponse>("betHistory", "/api/bet-history", league, EMPTY_BET_HISTORY);
  const gamesToday = useDashboardData<GamesTodayResponse>("gamesToday", "/api/games-today", league, EMPTY_GAMES_TODAY);

  const { policies, frontierPolicies, byKey } = useMemo(
    () => collectBetSizingPolicies(betHistory.data.strategy_configs, betHistory.data.strategy_optimization.frontier),
    [betHistory.data.strategy_configs, betHistory.data.strategy_optimization.frontier]
  );

  const defaultPolicyKey = useMemo(
    () => selectDefaultPolicyKey(strategy, betHistory.data.strategy_configs, frontierPolicies),
    [betHistory.data.strategy_configs, frontierPolicies, strategy]
  );

  const [selectedPolicyKey, setSelectedPolicyKey] = useState("");
  const [selectedGameId, setSelectedGameId] = useState<number | null>(null);
  const effectiveSelectedPolicyKey = selectedPolicyKey && byKey.has(selectedPolicyKey) ? selectedPolicyKey : defaultPolicyKey;
  const selectedPolicy = byKey.get(effectiveSelectedPolicyKey) || byKey.get(defaultPolicyKey) || policies[0] || null;
  const officialPolicyKey = betHistory.data.strategy_configs[strategy]?.config_signature || null;
  const officialPolicy = officialPolicyKey ? byKey.get(officialPolicyKey) || null : null;
  const replayRankingAvailable = frontierPolicies.length > 0;
  const slate = useMemo(() => selectBetSizingSlate(gamesToday.data), [gamesToday.data]);

  const gamePreviews = useMemo(
    () => (selectedPolicy ? buildBetSizingGamePreviews(slate.rows, strategy, selectedPolicy) : []),
    [selectedPolicy, slate.rows, strategy]
  );

  const defaultGameId = useMemo(() => selectDefaultGameId(gamePreviews), [gamePreviews]);
  const effectiveSelectedGameId =
    selectedGameId !== null && gamePreviews.some((preview) => preview.row.game_id === selectedGameId)
      ? selectedGameId
      : defaultGameId;

  const explainer = useMemo(
    () => (selectedPolicy ? buildBetSizingExplainerModel(gamePreviews, selectedPolicy, slate, effectiveSelectedGameId) : null),
    [effectiveSelectedGameId, gamePreviews, selectedPolicy, slate]
  );

  const selectedGame = explainer?.selectedGame || null;
  const loading = betHistory.isLoading || gamesToday.isLoading;
  const error = betHistory.error || gamesToday.error;

  return (
    <div className="grid">
      <section className={`card ${styles.heroCard}`}>
        <div className={styles.heroTop}>
          <div>
            <p className={styles.eyebrow}>Bet Sizing</p>
            <h2 className="title">{explainer?.headline || "How today&apos;s budget turns into a bet amount"}</h2>
            <p className={styles.heroText}>
              {explainer?.dek ||
                "This page starts with the daily budget, shows how each game earns or loses the right to ask for money, and then shows exactly how much of that budget lands on each game."}
            </p>
          </div>
          <div className={styles.heroMeta}>
            <span className={styles.heroPill}>{league}</span>
            <span className={styles.heroPill}>{getBetStrategyConfig(strategy).label}</span>
          </div>
        </div>

        {explainer ? (
          <>
            <div className={styles.heroMetrics}>
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricLabel}>Daily budget</span>
                <strong className={styles.heroMetricValue}>{formatUsd(explainer.totalBudget)}</strong>
              </div>
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricLabel}>Max per game</span>
                <strong className={styles.heroMetricValue}>{formatUsd(explainer.maxBetSize)}</strong>
              </div>
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricLabel}>Budget committed</span>
                <strong className={styles.heroMetricValue}>{formatUsd(explainer.allocatedBudget)}</strong>
              </div>
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricLabel}>Budget left</span>
                <strong className={styles.heroMetricValue}>{formatUsd(explainer.remainingBudget)}</strong>
              </div>
            </div>

            <div className={styles.heroBudgetBar}>
              {explainer.allocationSteps
                .filter((step) => step.finalStake > 0)
                .map((step) => (
                  <span
                    key={step.gameId}
                    className={styles.heroBudgetSegment}
                    style={{ width: `${(step.finalStake / explainer.totalBudget) * 100}%` }}
                  />
                ))}
              {explainer.remainingBudget > 0 ? (
                <span
                  className={styles.heroBudgetRemainder}
                  style={{ width: `${(explainer.remainingBudget / explainer.totalBudget) * 100}%` }}
                />
              ) : null}
            </div>
          </>
        ) : null}

        <div className={styles.stageGrid}>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>1. Screen</p>
            <p className={styles.stageTitle}>Check whether the game deserves any money at all</p>
            <p className={styles.stageBody}>The app looks for priced games, positive value after adjustment, and a large enough edge.</p>
          </article>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>2. Size</p>
            <p className={styles.stageTitle}>Let the game ask for a stake</p>
            <p className={styles.stageBody}>A base bankroll fraction asks for an amount, then the profile scale and per-game cap cut it down.</p>
          </article>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>3. Allocate</p>
            <p className={styles.stageTitle}>Spend today&apos;s budget on the best asks first</p>
            <p className={styles.stageBody}>Higher-value bets receive budget first until the day&apos;s max risk is reached.</p>
          </article>
        </div>

        <div className={styles.asOfRow}>
          <span className="small">{explainer?.slateLabel || slate.label}</span>
          <span className="small">Replay data as of {formatAsOfLabel(gamesToday.data.as_of_utc)}</span>
          <span className="small">Odds snapshot as of {formatAsOfLabel(gamesToday.data.odds_as_of_utc)}</span>
        </div>
      </section>

      {loading ? <p className="small">Loading bet sizing view...</p> : null}
      {error ? <p className="small">Failed to load bet sizing data: {error}</p> : null}

      {!loading && !error && selectedPolicy && explainer ? (
        <>
          <section className={`card ${styles.policyCard}`}>
            <div className={styles.policyHeader}>
              <div>
                <p className={styles.eyebrow}>Stage 0</p>
                <h2 className="title">Pick the House Rules</h2>
                <p className="small">
                  The strategy changes the daily budget, the max size of any one bet, and whether underdogs are allowed.
                </p>
              </div>
              <div className={styles.policyButtonRow}>
                {BET_STRATEGIES.map((code) => {
                  const policy = betHistory.data.strategy_configs[code];
                  const isSelected = policy.config_signature === selectedPolicy.configSignature;
                  return (
                    <button
                      key={code}
                      type="button"
                      className={`${styles.policyButton} ${isSelected ? styles.policyButtonActive : ""}`}
                      onClick={() => setSelectedPolicyKey(policy.config_signature)}
                    >
                      <span className={styles.policyButtonLabel}>{policy.label}</span>
                      <span className={styles.policyButtonNote}>{sourceLabel(policy.optimization_source)}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className={styles.summaryGrid}>
              <article className={styles.ruleCard}>
                <p className={styles.ruleEyebrow}>Selected policy</p>
                <h3 className={styles.ruleTitle}>{selectedPolicy.label}</h3>
                <p className={styles.ruleBody}>{selectedPolicy.description}</p>
                <div className={styles.ruleGrid}>
                  <div className={styles.ruleTile}>
                    <span className={styles.ruleLabel}>Daily budget</span>
                    <strong className={styles.ruleValue}>{formatUsd(selectedPolicy.maxDailyUnits * BET_UNIT_DOLLARS)}</strong>
                  </div>
                  <div className={styles.ruleTile}>
                    <span className={styles.ruleLabel}>Per-game cap</span>
                    <strong className={styles.ruleValue}>{formatUsd(selectedPolicy.maxBetUnits * BET_UNIT_DOLLARS)}</strong>
                  </div>
                  <div className={styles.ruleTile}>
                    <span className={styles.ruleLabel}>Value floor</span>
                    <strong className={styles.ruleValue}>{formatProbabilityPoints(selectedPolicy.minEdge)}</strong>
                  </div>
                  <div className={styles.ruleTile}>
                    <span className={styles.ruleLabel}>Stake scale</span>
                    <strong className={styles.ruleValue}>{selectedPolicy.fractionalKelly.toFixed(2)}x</strong>
                  </div>
                </div>
                <p className={styles.ruleFootnote}>
                  Dollar amounts are anchored to a {formatUsd(REFERENCE_BANKROLL_DOLLARS)} reference bankroll. For example, a {formatBankrollShare(1)} position is {formatUsd(BET_UNIT_DOLLARS)}.
                </p>
                {selectedPolicy.optimizationSource === "static_fallback" ? (
                  <p className={styles.ruleFootnote}>
                    Replay coverage is still thin, so this policy is using fixed defaults rather than a replay-ranked profile.
                  </p>
                ) : null}
              </article>

              <article className={styles.screeningCard}>
                <p className={styles.ruleEyebrow}>Stage 1</p>
                <h3 className={styles.ruleTitle}>Today&apos;s Screening Funnel</h3>
                <p className={styles.ruleBody}>Each bar shows how many games are still alive after one more rule is applied.</p>
                <div className={styles.screeningSteps}>
                  {explainer.screening.map((step, index) => {
                    const previousCount = index === 0 ? step.count : (explainer.screening[index - 1]?.count ?? 0);
                    const width = explainer.screening[0]?.count ? (step.count / explainer.screening[0].count) * 100 : 0;
                    return (
                      <div key={step.key} className={styles.screeningRow}>
                        <div className={styles.screeningCopy}>
                          <span className={styles.screeningLabel}>{step.label}</span>
                          <span className={styles.screeningDesc}>{step.description}</span>
                        </div>
                        <div className={styles.screeningBarWrap}>
                          <div className={styles.screeningTrack}>
                            <span
                              className={`${styles.screeningBar} ${screeningTone(step.count, previousCount)}`}
                              style={{ width: `${width}%` }}
                            />
                          </div>
                          <strong className={styles.screeningCount}>{step.count}</strong>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </article>
            </div>
          </section>

          <BetSizingBudgetFlow
            league={league}
            totalBudget={explainer.totalBudget}
            allocatedBudget={explainer.allocatedBudget}
            remainingBudget={explainer.remainingBudget}
            steps={explainer.allocationSteps}
          />

          <section className={styles.explorerGrid}>
            <article className={`card ${styles.gamesCard}`}>
              <div className={styles.gamesHeader}>
                <div>
                  <p className={styles.eyebrow}>Slate Explorer</p>
                  <h2 className="title">Every Game, Bet or Pass</h2>
                  <p className="small">Pick a game to see the exact walk from price to final stake.</p>
                </div>
                <div className={styles.gamesPills}>
                  <span className={styles.heroPill}>{explainer.fundedBetCount} funded</span>
                  <span className={styles.heroPill}>{explainer.passCount} passed</span>
                </div>
              </div>

              <div className={styles.gameList}>
                {explainer.games.map((game) => {
                  const isSelected = game.preview.row.game_id === selectedGame?.preview.row.game_id;
                  return (
                    <button
                      key={game.preview.row.game_id}
                      type="button"
                      className={`${styles.gameButton} ${isSelected ? styles.gameButtonActive : ""}`}
                      onClick={() => setSelectedGameId(game.preview.row.game_id)}
                    >
                      <div className={styles.gameButtonTop}>
                        <span className={styles.tipBadge}>{formatTipTime(game.preview.row.start_time_utc)} CT</span>
                        <span className={`${styles.statusBadge} ${game.status === "bet" ? styles.statusBet : styles.statusPass}`}>
                          {game.status === "bet" ? formatUsd(game.finalStake) : game.passLabel}
                        </span>
                      </div>
                      <TeamMatchup
                        league={league}
                        awayTeamCode={game.preview.row.away_team}
                        homeTeamCode={game.preview.row.home_team}
                        awayLabel={game.preview.row.away_team}
                        homeLabel={game.preview.row.home_team}
                        size="sm"
                      />
                      <div className={styles.gameStats}>
                        <span>{game.allocationRank ? `Rank #${game.allocationRank}` : `Stops at ${game.stopStage}`}</span>
                        <span>{formatProbabilityPoints(game.preview.trace.candidateEdge)}</span>
                        <span>{formatExpectedValue(game.preview.trace.candidateExpectedValue)}</span>
                      </div>
                      <p className={styles.gameSummary}>{game.laymanSummary}</p>
                    </button>
                  );
                })}
              </div>
            </article>

            {selectedGame ? (
              <div className={styles.detailColumn}>
                <section className={`card ${styles.selectedCard}`}>
                  <div className={styles.selectedTop}>
                    <div>
                      <p className={styles.eyebrow}>Selected game</p>
                      <h2 className="title">{selectedGame.matchupLabel}</h2>
                      <p className="small">{selectedGame.laymanSummary}</p>
                    </div>
                    <BetStakeWithIcon
                      league={league}
                      teamCode={selectedGame.preview.trace.decision.team}
                      label={selectedGame.preview.trace.decision.team}
                      stake={selectedGame.finalStake}
                      size="md"
                      zeroLabel="$0 pass"
                      className={styles.breakdownStake}
                    />
                  </div>

                  <div className={styles.matchupDetails}>
                    <TeamMatchup
                      league={league}
                      awayTeamCode={selectedGame.preview.row.away_team}
                      homeTeamCode={selectedGame.preview.row.home_team}
                      awayLabel={selectedGame.preview.row.away_team}
                      homeLabel={selectedGame.preview.row.home_team}
                      size="md"
                    />
                    <div className={styles.factGrid}>
                      <div className={styles.factTile}>
                        <span className={styles.factLabel}>Home model win %</span>
                        <strong className={styles.factValue}>{formatPercent(selectedGame.preview.trace.homeRawModelProbability)}</strong>
                      </div>
                      <div className={styles.factTile}>
                        <span className={styles.factLabel}>Home moneyline</span>
                        <strong className={styles.factValue}>{formatMoneyline(selectedGame.preview.row.home_moneyline)}</strong>
                      </div>
                      <div className={styles.factTile}>
                        <span className={styles.factLabel}>Away moneyline</span>
                        <strong className={styles.factValue}>{formatMoneyline(selectedGame.preview.row.away_moneyline)}</strong>
                      </div>
                      <div className={styles.factTile}>
                        <span className={styles.factLabel}>Requested stake</span>
                        <strong className={styles.factValue}>{formatUsd(selectedGame.requestedStake)}</strong>
                      </div>
                    </div>
                  </div>
                </section>

                <BetSizingProbabilityBridge
                  marketProbability={selectedGame.preview.trace.candidateMarketProbability}
                  referenceProbability={selectedGame.preview.trace.candidateReferenceProbability}
                  adjustedProbability={selectedGame.preview.trace.candidateAdjustedProbability}
                  rawProbability={selectedGame.preview.trace.candidateRawModelProbability}
                  expectedValue={selectedGame.preview.trace.candidateExpectedValue}
                  edge={selectedGame.preview.trace.candidateEdge}
                />

                <BetSizingStakeFlow game={selectedGame} policy={selectedPolicy} sizingStyle="continuous" totalBudget={explainer.totalBudget} />

                <section className={`card ${styles.storyCard}`}>
                  <div>
                    <p className={styles.eyebrow}>Selected game summary</p>
                    <h3 className="title">Why this game ends where it does</h3>
                  </div>
                  <div className={styles.storyGrid}>
                    <div className={styles.storyTile}>
                      <span className={styles.storyLabel}>Chosen side</span>
                      <strong className={styles.storyValue}>{selectedGame.preview.trace.decision.team || "No bet"}</strong>
                    </div>
                    <div className={styles.storyTile}>
                      <span className={styles.storyLabel}>Adjusted edge</span>
                      <strong className={styles.storyValue}>{formatProbabilityPoints(selectedGame.preview.trace.candidateEdge)}</strong>
                    </div>
                    <div className={styles.storyTile}>
                      <span className={styles.storyLabel}>Expected value</span>
                      <strong className={styles.storyValue}>{formatExpectedValue(selectedGame.preview.trace.candidateExpectedValue)}</strong>
                    </div>
                    <div className={styles.storyTile}>
                      <span className={styles.storyLabel}>Scale after policy</span>
                      <strong className={styles.storyValue}>{formatUnits(selectedGame.scaledKellyUnits)}</strong>
                    </div>
                    <div className={styles.storyTile}>
                      <span className={styles.storyLabel}>Budget rank</span>
                      <strong className={styles.storyValue}>{selectedGame.allocationRank ? `#${selectedGame.allocationRank}` : "Not funded"}</strong>
                    </div>
                    <div className={styles.storyTile}>
                      <span className={styles.storyLabel}>Budget after this game</span>
                      <strong className={styles.storyValue}>
                        {selectedGame.budgetAfter !== null ? formatUsd(selectedGame.budgetAfter) : "—"}
                      </strong>
                    </div>
                  </div>
                </section>
              </div>
            ) : null}
          </section>

          <details className={`card ${styles.advancedCard}`}>
            <summary className={styles.advancedSummary}>Advanced replay diagnostics and formulas</summary>
            <div className={styles.advancedBody}>
              <div className={styles.advancedGrid}>
                <article className={styles.metricCard}>
                  <p className={styles.ruleEyebrow}>Replay notes</p>
                  <h3 className={styles.ruleTitle}>Policy backtest context</h3>
                  <p className={styles.ruleBody}>
                    {replayRankingAvailable
                      ? "Replay-tested points are available below. Use them to compare higher-risk and lower-risk policy shapes."
                      : "Replay policy ranking stays muted until more matched replay coverage is available."}
                  </p>
                  <div className={styles.ruleGrid}>
                    <div className={styles.ruleTile}>
                      <span className={styles.ruleLabel}>Candidates tested</span>
                      <strong className={styles.ruleValue}>{betHistory.data.strategy_optimization.candidate_count}</strong>
                    </div>
                    <div className={styles.ruleTile}>
                      <span className={styles.ruleLabel}>Replay points</span>
                      <strong className={styles.ruleValue}>{betHistory.data.strategy_optimization.frontier_point_count}</strong>
                    </div>
                    <div className={styles.ruleTile}>
                      <span className={styles.ruleLabel}>ROI</span>
                      <strong className={styles.ruleValue}>
                        {metricOrDash(selectedPolicy.metrics, (metrics) => metrics.roi, (value) => formatPercent(value))}
                      </strong>
                    </div>
                    <div className={styles.ruleTile}>
                      <span className={styles.ruleLabel}>Log growth / bet</span>
                      <strong className={styles.ruleValue}>
                        {metricOrDash(selectedPolicy.metrics, (metrics) => metrics.expected_log_growth_per_bet, (value) => value.toFixed(4))}
                      </strong>
                    </div>
                  </div>
                </article>

                <article className={styles.metricCard}>
                  <p className={styles.ruleEyebrow}>Exact formulas</p>
                  <h3 className={styles.ruleTitle}>Behind the walk-through</h3>
                  <div className={styles.formulaBody}>
                    <p className="small">Reference probability = 70% market fair probability + 30% peer-model consensus when peer models exist.</p>
                    <p className="small">Adjusted probability = reference probability + confidence weight × (raw model probability - reference probability).</p>
                    <p className="small">Edge = adjusted probability - market fair probability.</p>
                    <p className="small">Expected value = adjusted probability × decimal odds - 1.</p>
                    <p className="small">Base stake fraction = (adjusted probability × decimal odds - 1) / (decimal odds - 1).</p>
                    <p className="small">Final stake = scaled base amount, capped per game, then capped again by the daily budget.</p>
                  </div>
                </article>
              </div>

              <BetSizingFrontier
                points={frontierPolicies}
                selectedKey={selectedPolicy.configSignature}
                officialPolicy={officialPolicy}
                onSelect={setSelectedPolicyKey}
              />
            </div>
          </details>
        </>
      ) : null}
    </div>
  );
}
