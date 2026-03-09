import {
  BET_UNIT_DOLLARS,
  HISTORICAL_BANKROLL_START_DOLLARS,
  explainBetDecisionsForSlate,
  settleBet,
} from "@/lib/betting";
import type { ModelWinProbabilities } from "@/lib/betting-model";
import type { LeagueCode } from "@/lib/league";
import {
  BET_STRATEGIES,
  getBetStrategyConfig,
  toBetStrategyRuleConfig,
  type BetStrategy,
  type BetStrategyConfig,
  type BetStrategyRuleConfig,
} from "@/lib/betting-strategy";

export type OptimizableHistoricalBetRow = {
  game_id: number;
  league?: LeagueCode | null;
  date_central: string;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  home_moneyline: number;
  away_moneyline: number;
  home_win: number | null;
  betting_model_name?: string | null;
  model_win_probabilities?: ModelWinProbabilities | null;
};

export type BetStrategyPerformanceSnapshot = {
  mean_daily_profit_units: number;
  daily_volatility_units: number;
  downside_deviation_units: number;
  sharpe_ratio: number;
  expected_log_growth_per_bet: number;
  max_drawdown_units: number;
  total_profit: number;
  total_risked: number;
  roi: number;
  settled_bets: number;
  active_days: number;
  total_days: number;
};

export type ResolvedBetStrategyConfig = BetStrategyConfig & {
  config_signature: string;
  optimization_objective: string;
  optimization_source: "historical_frontier" | "historical_downside" | "static_fallback";
  metrics: BetStrategyPerformanceSnapshot | null;
};

export type FrontierPointSummary = BetStrategyPerformanceSnapshot &
  BetStrategyRuleConfig & {
    config_signature: string;
  };

export type BetStrategyOptimizationSummary = {
  method: string;
  sizing_style: "continuous";
  risk_free_rate: number;
  candidate_count: number;
  frontier_point_count: number;
  frontier: FrontierPointSummary[];
  selected: Record<BetStrategy, FrontierPointSummary | null>;
};

type CandidateEvaluation = {
  config: BetStrategyRuleConfig;
  configSignature: string;
  metrics: BetStrategyPerformanceSnapshot;
};

const OPTIMIZER_VERSION = "bet_strategy_frontier_v1";
const MIN_SETTLED_BETS = 3;
const MIN_ACTIVE_DAYS = 2;
const MIN_EDGE_GRID = [0.025, 0.03, 0.035];
const MIN_EXPECTED_VALUE_GRID = [0.015, 0.02, 0.025];
const FRACTIONAL_KELLY_GRID = [0.2, 0.25, 0.33, 0.5, 0.6, 0.75, 1];
const MAX_BET_UNITS_GRID = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
const MAX_DAILY_UNITS_GRID = [1.5, 2, 2.5, 3, 4, 5, 6];
const MIN_REPLAY_GAMES_FOR_POLICY_RANKING = 60;
const MIN_REPLAY_DAYS_FOR_POLICY_RANKING = 20;

function roundNumber(value: number, digits = 6): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function mean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleStandardDeviation(values: number[]): number {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const variance =
    values.reduce((sum, value) => {
      const diff = value - avg;
      return sum + diff * diff;
    }, 0) /
    (values.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}

function downsideDeviation(values: number[]): number {
  if (!values.length) return 0;
  const downsideSquares = values.reduce((sum, value) => {
    const downside = Math.min(0, value);
    return sum + downside * downside;
  }, 0);
  return Math.sqrt(downsideSquares / values.length);
}

function maxDrawdown(values: number[]): number {
  let peak = 0;
  let cumulative = 0;
  let maxObservedDrawdown = 0;

  for (const value of values) {
    cumulative += value;
    if (cumulative > peak) peak = cumulative;
    const drawdown = peak - cumulative;
    if (drawdown > maxObservedDrawdown) {
      maxObservedDrawdown = drawdown;
    }
  }

  return maxObservedDrawdown;
}

function expectedLogGrowthPerBet(dailyProfits: number[], settledBets: number): number {
  if (!dailyProfits.length || settledBets <= 0) return 0;

  let bankroll = HISTORICAL_BANKROLL_START_DOLLARS;
  let totalLogGrowth = 0;

  for (const profit of dailyProfits) {
    const nextBankroll = bankroll + profit;
    if (nextBankroll <= 0) {
      return Number.NEGATIVE_INFINITY;
    }
    totalLogGrowth += Math.log(nextBankroll / bankroll);
    bankroll = nextBankroll;
  }

  return totalLogGrowth / settledBets;
}

function strategyConfigSignature(config: BetStrategyRuleConfig): string {
  return [
    OPTIMIZER_VERSION,
    config.allowUnderdogs ? "dogs" : "favorites",
    roundNumber(config.minEdge, 3).toFixed(3),
    roundNumber(config.minExpectedValue, 3).toFixed(3),
    roundNumber(config.fractionalKelly, 3).toFixed(3),
    roundNumber(config.maxBetUnits, 3).toFixed(3),
    roundNumber(config.maxDailyUnits, 3).toFixed(3),
  ].join("|");
}

function strategyConfigSummary(config: BetStrategyRuleConfig, metrics: BetStrategyPerformanceSnapshot): FrontierPointSummary {
  return {
    config_signature: strategyConfigSignature(config),
    allowUnderdogs: config.allowUnderdogs,
    minEdge: config.minEdge,
    minExpectedValue: config.minExpectedValue,
    fractionalKelly: config.fractionalKelly,
    maxBetUnits: config.maxBetUnits,
    maxDailyUnits: config.maxDailyUnits,
    ...metrics,
  };
}

function buildCandidateGrid(): BetStrategyRuleConfig[] {
  const candidates: BetStrategyRuleConfig[] = [];

  for (const allowUnderdogs of [true, false]) {
    for (const minEdge of MIN_EDGE_GRID) {
      for (const minExpectedValue of MIN_EXPECTED_VALUE_GRID) {
        for (const fractionalKelly of FRACTIONAL_KELLY_GRID) {
          for (const maxBetUnits of MAX_BET_UNITS_GRID) {
            for (const maxDailyUnits of MAX_DAILY_UNITS_GRID) {
              candidates.push({
                allowUnderdogs,
                minEdge,
                minExpectedValue,
                fractionalKelly,
                maxBetUnits,
                maxDailyUnits,
              });
            }
          }
        }
      }
    }
  }

  return candidates;
}

function summarizeReplayCoverage(rows: OptimizableHistoricalBetRow[]): { settledGames: number; replayDays: number } {
  const settledRows = rows.filter((row) => row.home_win !== null);
  const replayDays = new Set(settledRows.map((row) => row.date_central).filter(Boolean)).size;
  return {
    settledGames: settledRows.length,
    replayDays,
  };
}

function replayCoverageGateMessage(settledGames: number, replayDays: number): string {
  return `Static defaults active until matched replay coverage reaches at least ${MIN_REPLAY_GAMES_FOR_POLICY_RANKING} settled games across ${MIN_REPLAY_DAYS_FOR_POLICY_RANKING} replay days (current sample: ${settledGames} games across ${replayDays} days).`;
}

function evaluateCandidate(rows: OptimizableHistoricalBetRow[], config: BetStrategyRuleConfig): CandidateEvaluation | null {
  const dailyProfitByDate = new Map<string, number>();
  let totalProfit = 0;
  let totalRisked = 0;
  let settledBets = 0;

  const rowsByDate = new Map<string, OptimizableHistoricalBetRow[]>();
  for (const row of rows) {
    if (row.home_win === null) continue;
    const current = rowsByDate.get(row.date_central) || [];
    current.push(row);
    rowsByDate.set(row.date_central, current);
  }

  for (const [dateCentral, dayRows] of rowsByDate.entries()) {
    const traces = explainBetDecisionsForSlate(
      dayRows.map((row) => ({
        home_team: row.home_team,
        away_team: row.away_team,
        home_win_probability: row.home_win_probability,
        home_moneyline: row.home_moneyline,
        away_moneyline: row.away_moneyline,
        betting_model_name: row.betting_model_name,
        model_win_probabilities: row.model_win_probabilities,
      })),
      "riskAdjusted",
      "continuous",
      config
    );

    dailyProfitByDate.set(dateCentral, 0);

    for (const [index, trace] of traces.entries()) {
      const settlement = settleBet(trace.decision, dayRows[index]?.home_win);
      if (settlement.outcome === "no_bet") continue;

      dailyProfitByDate.set(dateCentral, (dailyProfitByDate.get(dateCentral) || 0) + settlement.profit);
      totalProfit += settlement.profit;
      totalRisked += trace.decision.stake;
      settledBets += 1;
    }
  }

  const dailyProfitEntries = Array.from(dailyProfitByDate.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, profit]) => profit);
  const dailyProfitUnits = dailyProfitEntries.map((profit) => profit / BET_UNIT_DOLLARS);

  const activeDays = dailyProfitUnits.filter((value) => value !== 0).length;
  const dailyVolatilityUnits = sampleStandardDeviation(dailyProfitUnits);
  const meanDailyProfitUnits = mean(dailyProfitUnits);
  const sharpeRatio = dailyVolatilityUnits > 0 ? meanDailyProfitUnits / dailyVolatilityUnits : 0;
  const logGrowth = expectedLogGrowthPerBet(dailyProfitEntries, settledBets);

  const metrics: BetStrategyPerformanceSnapshot = {
    mean_daily_profit_units: roundNumber(meanDailyProfitUnits),
    daily_volatility_units: roundNumber(dailyVolatilityUnits),
    downside_deviation_units: roundNumber(downsideDeviation(dailyProfitUnits)),
    sharpe_ratio: roundNumber(sharpeRatio),
    expected_log_growth_per_bet: roundNumber(logGrowth),
    max_drawdown_units: roundNumber(maxDrawdown(dailyProfitUnits)),
    total_profit: roundNumber(totalProfit),
    total_risked: roundNumber(totalRisked),
    roi: roundNumber(totalRisked > 0 ? totalProfit / totalRisked : 0),
    settled_bets: settledBets,
    active_days: activeDays,
    total_days: dailyProfitUnits.length,
  };

  if (
    settledBets < MIN_SETTLED_BETS ||
    activeDays < MIN_ACTIVE_DAYS ||
    totalRisked <= 0 ||
    dailyProfitUnits.length < 2 ||
    !Number.isFinite(metrics.daily_volatility_units)
  ) {
    return null;
  }

  return {
    config,
    configSignature: strategyConfigSignature(config),
    metrics,
  };
}

function buildEfficientFrontier(candidates: CandidateEvaluation[]): CandidateEvaluation[] {
  const sorted = [...candidates].sort(
    (left, right) =>
      left.metrics.daily_volatility_units - right.metrics.daily_volatility_units ||
      right.metrics.mean_daily_profit_units - left.metrics.mean_daily_profit_units ||
      right.metrics.sharpe_ratio - left.metrics.sharpe_ratio
  );

  const frontier: CandidateEvaluation[] = [];
  let bestMean = Number.NEGATIVE_INFINITY;

  for (const candidate of sorted) {
    if (candidate.metrics.mean_daily_profit_units <= bestMean) {
      continue;
    }
    frontier.push(candidate);
    bestMean = candidate.metrics.mean_daily_profit_units;
  }

  return frontier;
}

function compareBySharpe(left: CandidateEvaluation, right: CandidateEvaluation): number {
  return (
    right.metrics.expected_log_growth_per_bet - left.metrics.expected_log_growth_per_bet ||
    right.metrics.sharpe_ratio - left.metrics.sharpe_ratio ||
    right.metrics.mean_daily_profit_units - left.metrics.mean_daily_profit_units ||
    left.metrics.daily_volatility_units - right.metrics.daily_volatility_units
  );
}

function compareByReturn(left: CandidateEvaluation, right: CandidateEvaluation): number {
  return (
    right.metrics.mean_daily_profit_units - left.metrics.mean_daily_profit_units ||
    right.metrics.total_profit - left.metrics.total_profit ||
    right.metrics.sharpe_ratio - left.metrics.sharpe_ratio
  );
}

function compareByCapitalProtection(left: CandidateEvaluation, right: CandidateEvaluation): number {
  return (
    left.metrics.downside_deviation_units - right.metrics.downside_deviation_units ||
    left.metrics.max_drawdown_units - right.metrics.max_drawdown_units ||
    left.metrics.daily_volatility_units - right.metrics.daily_volatility_units ||
    right.metrics.mean_daily_profit_units - left.metrics.mean_daily_profit_units
  );
}

function toResolvedConfig(
  strategy: BetStrategy,
  candidate: CandidateEvaluation | null,
  optimizationSource: ResolvedBetStrategyConfig["optimization_source"],
  optimizationObjective: string
): ResolvedBetStrategyConfig {
  const fallback = getBetStrategyConfig(strategy);
  const resolvedRules = candidate?.config || toBetStrategyRuleConfig(fallback);

  return {
    ...fallback,
    ...resolvedRules,
    config_signature: candidate?.configSignature || strategyConfigSignature(resolvedRules),
    optimization_objective: optimizationObjective,
    optimization_source: optimizationSource,
    metrics: candidate?.metrics || null,
  };
}

export function resolveBetStrategyConfigs(rows: OptimizableHistoricalBetRow[]): {
  strategyConfigs: Record<BetStrategy, ResolvedBetStrategyConfig>;
  optimizationSummary: BetStrategyOptimizationSummary;
} {
  const replayCoverage = summarizeReplayCoverage(rows);
  const hasReplayCoverageForRanking =
    replayCoverage.settledGames >= MIN_REPLAY_GAMES_FOR_POLICY_RANKING &&
    replayCoverage.replayDays >= MIN_REPLAY_DAYS_FOR_POLICY_RANKING;

  if (!hasReplayCoverageForRanking) {
    return {
      strategyConfigs: {
        riskAdjusted: toResolvedConfig(
          "riskAdjusted",
          null,
          "static_fallback",
          "Static balanced default while matched replay coverage remains limited"
        ),
        aggressive: toResolvedConfig(
          "aggressive",
          null,
          "static_fallback",
          "Static aggressive default while matched replay coverage remains limited"
        ),
        capitalPreservation: toResolvedConfig(
          "capitalPreservation",
          null,
          "static_fallback",
          "Static conservative default while matched replay coverage remains limited"
        ),
      },
      optimizationSummary: {
        method: replayCoverageGateMessage(replayCoverage.settledGames, replayCoverage.replayDays),
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
    };
  }

  const evaluatedCandidates = buildCandidateGrid()
    .map((config) => evaluateCandidate(rows, config))
    .filter((candidate): candidate is CandidateEvaluation => candidate !== null);

  const positiveMeanCandidates = evaluatedCandidates.filter(
    (candidate) =>
      candidate.metrics.mean_daily_profit_units > 0 && candidate.metrics.expected_log_growth_per_bet > Number.NEGATIVE_INFINITY
  );
  const frontier = buildEfficientFrontier(positiveMeanCandidates);
  const bestRiskAdjustedCandidate = [...frontier].sort(compareBySharpe)[0] || null;

  const aggressivePool = frontier.filter(
    (candidate) =>
      bestRiskAdjustedCandidate !== null &&
      candidate.configSignature !== bestRiskAdjustedCandidate.configSignature &&
      candidate.metrics.mean_daily_profit_units >= bestRiskAdjustedCandidate.metrics.mean_daily_profit_units &&
      candidate.metrics.daily_volatility_units >= bestRiskAdjustedCandidate.metrics.daily_volatility_units
  );
  const aggressiveCandidate =
    [...aggressivePool].sort(compareByReturn)[0] ||
    [...positiveMeanCandidates]
      .filter(
        (candidate) => bestRiskAdjustedCandidate !== null && candidate.configSignature !== bestRiskAdjustedCandidate.configSignature
      )
      .sort(compareByReturn)[0] ||
    null;

  const capitalPreservationCandidate =
    [...positiveMeanCandidates]
      .filter((candidate) => candidate.config.allowUnderdogs === false)
      .sort(compareByCapitalProtection)[0] || null;

  const selected: Record<BetStrategy, CandidateEvaluation | null> = {
    riskAdjusted: bestRiskAdjustedCandidate,
    aggressive: aggressiveCandidate,
    capitalPreservation: capitalPreservationCandidate,
  };

  const strategyConfigs: Record<BetStrategy, ResolvedBetStrategyConfig> = {
    riskAdjusted: toResolvedConfig(
      "riskAdjusted",
      bestRiskAdjustedCandidate,
      bestRiskAdjustedCandidate ? "historical_frontier" : "static_fallback",
      "Best replay expected log growth among sampled policies"
    ),
    aggressive: toResolvedConfig(
      "aggressive",
      aggressiveCandidate,
      aggressiveCandidate ? "historical_frontier" : "static_fallback",
      "Higher-return replay policy than the balanced selection"
    ),
    capitalPreservation: toResolvedConfig(
      "capitalPreservation",
      capitalPreservationCandidate,
      capitalPreservationCandidate ? "historical_downside" : "static_fallback",
      "Minimum downside volatility and drawdown among positive-return conservative replay candidates"
    ),
  };

  return {
    strategyConfigs,
      optimizationSummary: {
      method: "Historical replay candidate sweep ranked by expected log growth, return, and drawdown",
      sizing_style: "continuous",
      risk_free_rate: 0,
      candidate_count: evaluatedCandidates.length,
      frontier_point_count: frontier.length,
      frontier: frontier.map((candidate) => strategyConfigSummary(candidate.config, candidate.metrics)),
      selected: BET_STRATEGIES.reduce((acc, strategy) => {
        const candidate = selected[strategy];
        acc[strategy] = candidate ? strategyConfigSummary(candidate.config, candidate.metrics) : null;
        return acc;
      }, {} as Record<BetStrategy, FrontierPointSummary | null>),
    },
  };
}
