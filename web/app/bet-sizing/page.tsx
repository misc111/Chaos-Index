"use client";

import { Suspense, useMemo, useState } from "react";
import BetSizingFrontier from "@/components/BetSizingFrontier";
import { BetStakeWithIcon, TeamMatchup } from "@/components/TeamWithIcon";
import { BET_STRATEGIES, getBetSizingStyleConfig, getBetStrategyConfig } from "@/lib/betting-strategy";
import type { BetStrategyPerformanceSnapshot, FrontierPointSummary, ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import type { BetHistoryResponse, BetHistorySizingBundle, BetHistoryStrategyBundle } from "@/lib/bet-history-types";
import {
  buildBetSizingGamePreviews,
  collectBetSizingPolicies,
  selectBetSizingSlate,
  selectDefaultGameId,
  selectDefaultPolicyKey,
  type BetSizingGamePreview,
  type BetSizingPolicyPreview,
} from "@/lib/bet-sizing-view";
import {
  BET_UNIT_BANKROLL_FRACTION,
  BET_UNIT_DOLLARS,
  HISTORICAL_BANKROLL_START_DATE_CENTRAL,
  HISTORICAL_BANKROLL_START_DOLLARS,
  REFERENCE_BANKROLL_DOLLARS,
} from "@/lib/betting";
import { formatUsd } from "@/lib/currency";
import { useBetSizingStyle } from "@/lib/hooks/useBetSizingStyle";
import { useBetStrategy } from "@/lib/hooks/useBetStrategy";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { GamesTodayResponse } from "@/lib/types";
import styles from "./styles.module.css";

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatUnits(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)}u`;
}

function formatLogGrowth(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(4);
}

function formatProbabilityPoints(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const points = value * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pts`;
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

function sourceLabel(value: BetSizingPolicyPreview["optimizationSource"]): string {
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

function gateTone(passed: boolean): string {
  return passed ? styles.gatePass : styles.gateFail;
}

function buildEmptyStrategyConfigs(): Record<"riskAdjusted" | "aggressive" | "capitalPreservation", ResolvedBetStrategyConfig> {
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

function buildEmptySizingBundle(): BetHistorySizingBundle {
  return {
    continuous: EMPTY_BET_HISTORY_STRATEGY,
    bucketed: EMPTY_BET_HISTORY_STRATEGY,
  };
}

const EMPTY_BET_HISTORY: BetHistoryResponse = {
  league: "NHL",
  default_strategy: "riskAdjusted",
  default_sizing_style: "continuous",
  strategy_configs: buildEmptyStrategyConfigs(),
  strategy_optimization: {
    method: "",
    sizing_style: "continuous",
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
    riskAdjusted: buildEmptySizingBundle(),
    aggressive: buildEmptySizingBundle(),
    capitalPreservation: buildEmptySizingBundle(),
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

function BetSizingPageContent() {
  const league = useLeague();
  const strategy = useBetStrategy();
  const sizingStyle = useBetSizingStyle();
  const sizingStyleConfig = getBetSizingStyleConfig(sizingStyle);

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

  const [selectedPolicyKey, setSelectedPolicyKey] = useState<string>("");
  const [selectedGameId, setSelectedGameId] = useState<number | null>(null);
  const effectiveSelectedPolicyKey = selectedPolicyKey && byKey.has(selectedPolicyKey) ? selectedPolicyKey : defaultPolicyKey;
  const selectedPolicy = byKey.get(effectiveSelectedPolicyKey) || byKey.get(defaultPolicyKey) || policies[0] || null;
  const officialPolicyKey = betHistory.data.strategy_configs[strategy]?.config_signature || null;
  const officialPolicy = officialPolicyKey ? byKey.get(officialPolicyKey) || null : null;
  const replayRankingAvailable = frontierPolicies.length > 0;
  const slate = useMemo(() => selectBetSizingSlate(gamesToday.data), [gamesToday.data]);

  const gamePreviews = useMemo(
    () => (selectedPolicy ? buildBetSizingGamePreviews(slate.rows, strategy, sizingStyle, selectedPolicy) : []),
    [selectedPolicy, sizingStyle, slate.rows, strategy]
  );

  const defaultGameId = useMemo(() => selectDefaultGameId(gamePreviews), [gamePreviews]);
  const effectiveSelectedGameId =
    selectedGameId !== null && gamePreviews.some((preview) => preview.row.game_id === selectedGameId)
      ? selectedGameId
      : defaultGameId;
  const selectedGame =
    gamePreviews.find((preview) => preview.row.game_id === effectiveSelectedGameId) ||
    gamePreviews[0] ||
    (null as BetSizingGamePreview | null);

  const loading = betHistory.isLoading || gamesToday.isLoading;
  const error = betHistory.error || gamesToday.error;

  return (
    <div className="grid">
      <section className={`card ${styles.heroCard}`}>
        <div className={styles.heroTop}>
          <div>
            <p className={styles.eyebrow}>Bet Sizing</p>
            <h2 className="title">How the App Picks a Bet Amount</h2>
            <p className={styles.heroText}>
              First we choose a house policy. If replay coverage is deep enough, the app can re-rank those policies; otherwise it sticks with the defaults. Then we turn that style into a dollar amount for each game.
            </p>
          </div>
          <div className={styles.heroMeta}>
            <span className={styles.heroPill}>{league}</span>
            <span className={styles.heroPill}>{getBetStrategyConfig(strategy).label}</span>
            <span className={styles.heroPill}>{sizingStyleConfig.label}</span>
          </div>
        </div>

        <div className={styles.stageGrid}>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>1. Choose</p>
            <p className={styles.stageTitle}>Pick an overall risk style</p>
            <p className={styles.stageBody}>Risk styles now share one value screen. They mainly differ by stake scale, per-bet caps, and daily risk budget.</p>
          </article>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>2. Price</p>
            <p className={styles.stageTitle}>Compare model vs market</p>
            <p className={styles.stageBody}>The app shrinks the raw model probability toward market fair odds and peer-model consensus, then only keeps edges that still clear the shared floors.</p>
          </article>
          <article className={styles.stageCard}>
            <p className={styles.stageStep}>3. Size</p>
            <p className={styles.stageTitle}>Scale the final stake</p>
            <p className={styles.stageBody}>
              Stakes use an edge-scaled bankroll fraction on a ${REFERENCE_BANKROLL_DOLLARS.toLocaleString()} reference bankroll, cap each bet in units, then enforce the daily risk budget.
            </p>
          </article>
        </div>

        <div className={styles.asOfRow}>
          <span className="small">Replay data as of {formatAsOfLabel(gamesToday.data.as_of_utc)}</span>
          <span className="small">Odds snapshot as of {formatAsOfLabel(gamesToday.data.odds_as_of_utc)}</span>
        </div>
      </section>

      {loading ? <p className="small">Loading bet sizing view...</p> : null}
      {error ? <p className="small">Failed to load bet sizing data: {error}</p> : null}

      {!loading && !error && selectedPolicy ? (
        <>
          <section className={`card ${styles.policyChooser}`}>
            <div>
              <p className={styles.eyebrow}>Saved Policies</p>
              <h2 className="title">Official Policies</h2>
              <p className="small">
                {replayRankingAvailable
                  ? "Start with the saved policies, then click any replay point below to preview a different tradeoff."
                  : "Replay coverage is still too thin to rank alternative policies confidently, so the app is using the saved defaults for now."}
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
          </section>

          <BetSizingFrontier
            points={frontierPolicies}
            selectedKey={selectedPolicy.configSignature}
            officialPolicy={officialPolicy}
            onSelect={setSelectedPolicyKey}
          />

          <section className="grid two">
            <article className={`card ${styles.metricCard}`}>
              <p className={styles.eyebrow}>Selected Policy</p>
              <h2 className="title">{selectedPolicy.label}</h2>
              <p className="small">{selectedPolicy.description}</p>
              {selectedPolicy.optimizationSource === "static_fallback" ? (
                <p className="small">Matched replay coverage is still limited, so this profile is currently using its fixed default rules.</p>
              ) : null}

              <div className={styles.metricGrid}>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Mean daily profit</span>
                  <span className={styles.metricValue}>
                    {metricOrDash(selectedPolicy.metrics, (metrics) => metrics.mean_daily_profit_units, formatUnits)}
                  </span>
                </div>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Daily volatility</span>
                  <span className={styles.metricValue}>
                    {metricOrDash(selectedPolicy.metrics, (metrics) => metrics.daily_volatility_units, formatUnits)}
                  </span>
                </div>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Log growth / bet</span>
                  <span className={styles.metricValue}>
                    {metricOrDash(selectedPolicy.metrics, (metrics) => metrics.expected_log_growth_per_bet, formatLogGrowth)}
                  </span>
                </div>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>ROI</span>
                  <span className={styles.metricValue}>
                    {metricOrDash(selectedPolicy.metrics, (metrics) => metrics.roi, (value) => formatPercent(value))}
                  </span>
                </div>
              </div>

              <div className={styles.ruleGrid}>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Underdogs</span>
                  <span className={styles.ruleValue}>{selectedPolicy.allowUnderdogs ? "Allowed" : "Skipped"}</span>
                </div>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Min edge</span>
                  <span className={styles.ruleValue}>{formatProbabilityPoints(selectedPolicy.minEdge)}</span>
                </div>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Min EV</span>
                  <span className={styles.ruleValue}>{formatExpectedValue(selectedPolicy.minExpectedValue)}</span>
                </div>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Stake scale</span>
                  <span className={styles.ruleValue}>{selectedPolicy.fractionalKelly.toFixed(2)}x</span>
                </div>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Max bet size</span>
                  <span className={styles.ruleValue}>{selectedPolicy.maxBetUnits.toFixed(2)} units</span>
                </div>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Daily risk budget</span>
                  <span className={styles.ruleValue}>{selectedPolicy.maxDailyUnits.toFixed(2)} units</span>
                </div>
                <div className={styles.ruleRow}>
                  <span className={styles.ruleLabel}>Status</span>
                  <span className={styles.ruleValue}>{sourceLabel(selectedPolicy.optimizationSource)}</span>
                </div>
              </div>
            </article>

            <article className={`card ${styles.metricCard}`}>
              <p className={styles.eyebrow}>Sample Notes</p>
              <h2 className="title">Why Replay Is Only One Input</h2>
              <p className="small">
                {replayRankingAvailable
                  ? "The replay map is a provisional replay view. Expected log growth, ROI, and drawdown are more meaningful than a single Sharpe-style number, especially on sparse coverage."
                  : "Replay policy ranking stays de-emphasized until the app has enough matched odds history to compare bankroll paths with more confidence."}
              </p>
              <p className="small">{slate.label}</p>
              <div className={styles.metricGrid}>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Candidates tested</span>
                  <span className={styles.metricValue}>{betHistory.data.strategy_optimization.candidate_count}</span>
                </div>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Replay points</span>
                  <span className={styles.metricValue}>{betHistory.data.strategy_optimization.frontier_point_count}</span>
                </div>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Current mode</span>
                  <span className={styles.metricValue}>{sizingStyleConfig.label}</span>
                </div>
                <div className={styles.metricTile}>
                  <span className={styles.metricLabel}>Base unit</span>
                  <span className={styles.metricValue}>{formatUsd(BET_UNIT_DOLLARS)}</span>
                </div>
              </div>
              <p className="small">
                One unit is {formatUsd(BET_UNIT_DOLLARS)} because the sizing model uses a {formatUsd(REFERENCE_BANKROLL_DOLLARS)} reference bankroll and defines 1u as {(BET_UNIT_BANKROLL_FRACTION * 100).toFixed(0)}% of bankroll.
              </p>

              <details className={styles.formulaCard}>
                <summary>Show the exact sizing formulas</summary>
                <div className={styles.formulaBody}>
                  <p className="small">Reference probability = 70% market fair probability + 30% peer-model consensus (when peer models exist)</p>
                  <p className="small">Adjusted probability = reference probability + confidence weight × (raw model probability - reference probability)</p>
                  <p className="small">Edge = adjusted probability - market fair probability</p>
                  <p className="small">EV = adjusted probability × decimal odds - 1</p>
                  <p className="small">Base stake fraction = (adjusted probability × decimal odds - 1) / (decimal odds - 1)</p>
                  <p className="small">
                    Stake = round(min(max bet units × ${BET_UNIT_DOLLARS}, stake scale × base stake fraction × ${REFERENCE_BANKROLL_DOLLARS.toLocaleString()}))
                  </p>
                  <p className="small">The slate then trims lower-ranked bets if the day would exceed the selected daily risk budget.</p>
                  <p className="small">Bucketed mode rounds the continuous result into $0, $50, $100, or $150.</p>
                </div>
              </details>
            </article>
          </section>

          <section className={`card ${styles.gamesCard}`}>
            <div className={styles.gamesHeader}>
              <div>
                <p className={styles.eyebrow}>Stage 2</p>
                <h2 className="title">Preview the Slate Under This Policy</h2>
                <p className="small">Click a game to see exactly why it becomes a bet or a pass.</p>
              </div>
              <span className={styles.heroPill}>{slate.source === "upcoming" ? "Upcoming slate" : "Replay slate"}</span>
            </div>

            {!gamePreviews.length ? <p className="small">No games are available to preview yet.</p> : null}

            {gamePreviews.length ? (
              <div className={styles.gameGrid}>
                {gamePreviews.map((preview) => {
                  const isSelected = preview.row.game_id === selectedGame?.row.game_id;
                  const stakeTone = preview.trace.finalStake > 0 ? styles.stakePositive : styles.stakeNeutral;
                  return (
                    <button
                      key={preview.row.game_id}
                      type="button"
                      className={`${styles.gameButton} ${isSelected ? styles.gameButtonActive : ""}`}
                      onClick={() => setSelectedGameId(preview.row.game_id)}
                    >
                      <div className={styles.gameButtonTop}>
                        <span className={styles.tipBadge}>{formatTipTime(preview.row.start_time_utc)} CT</span>
                        <span className={`${styles.stakeBadge} ${stakeTone}`}>
                          {preview.trace.finalStake > 0 ? "Bet" : "Pass"}
                        </span>
                      </div>
                      <TeamMatchup
                        league={league}
                        awayTeamCode={preview.row.away_team}
                        homeTeamCode={preview.row.home_team}
                        awayLabel={preview.row.away_team}
                        homeLabel={preview.row.home_team}
                        size="md"
                        className={styles.matchup}
                      />
                      <div className={styles.gameButtonStats}>
                        <span>{preview.trace.finalStake > 0 ? formatUsd(preview.trace.finalStake) : "$0"}</span>
                        <span>{formatProbabilityPoints(preview.trace.candidateEdge)}</span>
                        <span>{formatExpectedValue(preview.trace.candidateExpectedValue)}</span>
                      </div>
                      <p className={styles.gameButtonReason}>{preview.trace.decision.reason}</p>
                    </button>
                  );
                })}
              </div>
            ) : null}
          </section>

          {selectedGame ? (
            <section className={`card ${styles.breakdownCard}`}>
              <div className={styles.breakdownHeader}>
                <div>
                  <p className={styles.eyebrow}>Stage 3</p>
                  <h2 className="title">Why This Game Becomes {selectedGame.trace.finalStake > 0 ? formatUsd(selectedGame.trace.finalStake) : "a Pass"}</h2>
                  <p className="small">{selectedGame.trace.decision.reason}</p>
                </div>
                <BetStakeWithIcon
                  league={league}
                  teamCode={selectedGame.trace.decision.team}
                  label={selectedGame.trace.decision.team}
                  stake={selectedGame.trace.finalStake}
                  size="md"
                  zeroLabel="$0 pass"
                  className={styles.breakdownStake}
                />
              </div>

              <div className={styles.breakdownTop}>
                <div className={styles.selectedMatchupCard}>
                  <TeamMatchup
                    league={league}
                    awayTeamCode={selectedGame.row.away_team}
                    homeTeamCode={selectedGame.row.home_team}
                    awayLabel={selectedGame.row.away_team}
                    homeLabel={selectedGame.row.home_team}
                    size="md"
                  />
                  <div className={styles.lineRow}>
                    <span>Home model win %</span>
                    <strong>{formatPercent(selectedGame.trace.homeRawModelProbability)}</strong>
                  </div>
                  <div className={styles.lineRow}>
                    <span>Home moneyline</span>
                    <strong>{formatMoneyline(selectedGame.row.home_moneyline)}</strong>
                  </div>
                  <div className={styles.lineRow}>
                    <span>Away moneyline</span>
                    <strong>{formatMoneyline(selectedGame.row.away_moneyline)}</strong>
                  </div>
                </div>

                <div className={styles.gateGrid}>
                  <div className={`${styles.gateChip} ${gateTone(selectedGame.trace.gates.positiveExpectedValue)}`}>Positive EV somewhere</div>
                  <div className={`${styles.gateChip} ${gateTone(selectedGame.trace.gates.edge)}`}>Edge clears floor</div>
                  <div className={`${styles.gateChip} ${gateTone(selectedGame.trace.gates.expectedValue)}`}>EV clears floor</div>
                  <div className={`${styles.gateChip} ${gateTone(selectedGame.trace.gates.underdogAllowed)}`}>Underdog rule</div>
                  <div className={`${styles.gateChip} ${gateTone(selectedGame.trace.gates.dailyBudget)}`}>Daily budget</div>
                </div>
              </div>

              <div className={styles.storyGrid}>
                <article className={styles.storyCard}>
                  <p className={styles.storyStep}>A. Compare</p>
                  <h3 className={styles.storyTitle}>Model vs market</h3>
                  <div className={styles.storyRows}>
                    <div className={styles.storyRow}>
                      <span>Chosen side</span>
                      <strong>{selectedGame.trace.decision.team || "No side"}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Raw model win probability</span>
                      <strong>{formatPercent(selectedGame.trace.candidateRawModelProbability)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Adjusted win probability</span>
                      <strong>{formatPercent(selectedGame.trace.candidateAdjustedProbability)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Reference probability</span>
                      <strong>{formatPercent(selectedGame.trace.candidateReferenceProbability)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Market fair probability</span>
                      <strong>{formatPercent(selectedGame.trace.candidateMarketProbability)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Edge</span>
                      <strong>{formatProbabilityPoints(selectedGame.trace.candidateEdge)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Expected value</span>
                      <strong>{formatExpectedValue(selectedGame.trace.candidateExpectedValue)}</strong>
                    </div>
                  </div>
                </article>

                <article className={styles.storyCard}>
                  <p className={styles.storyStep}>B. Scale</p>
                  <h3 className={styles.storyTitle}>Base stake signal with policy controls</h3>
                  <div className={styles.storyRows}>
                    <div className={styles.storyRow}>
                      <span>Base stake fraction</span>
                      <strong>{formatExpectedValue(selectedGame.trace.kellyFraction)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Base stake units</span>
                      <strong>{formatUnits(selectedGame.trace.rawKellyUnits)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Capped units</span>
                      <strong>{formatUnits(selectedGame.trace.cappedKellyUnits)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Stake scale</span>
                      <strong>{selectedPolicy.fractionalKelly.toFixed(2)}x</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Max bet size</span>
                      <strong>{selectedPolicy.maxBetUnits.toFixed(2)} units</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Daily risk budget</span>
                      <strong>{selectedPolicy.maxDailyUnits.toFixed(2)} units</strong>
                    </div>
                  </div>
                </article>

                <article className={styles.storyCard}>
                  <p className={styles.storyStep}>C. Finish</p>
                  <h3 className={styles.storyTitle}>Dollar amount</h3>
                  <div className={styles.storyRows}>
                    <div className={styles.storyRow}>
                      <span>Continuous amount</span>
                      <strong>{formatUsd(selectedGame.trace.continuousStake)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Bucketed amount</span>
                      <strong>{formatUsd(selectedGame.trace.bucketedStake)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Current sizing mode</span>
                      <strong>{sizingStyleConfig.label}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Pre-cap stake</span>
                      <strong>{formatUsd(selectedGame.trace.preDailyCapStake ?? selectedGame.trace.finalStake)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Final stake</span>
                      <strong>{formatUsd(selectedGame.trace.finalStake)}</strong>
                    </div>
                    <div className={styles.storyRow}>
                      <span>Chosen team</span>
                      <strong>{selectedGame.trace.decision.team || "No bet"}</strong>
                    </div>
                  </div>
                </article>
              </div>
            </section>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export default function BetSizingPage() {
  return (
    <Suspense fallback={<p className="small">Loading bet sizing view...</p>}>
      <BetSizingPageContent />
    </Suspense>
  );
}
