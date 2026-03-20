import type { LeagueCode } from "@/lib/league";

export type BetStrategy = "riskAdjusted" | "aggressive" | "capitalPreservation";

export const BET_STRATEGIES = ["riskAdjusted", "aggressive", "capitalPreservation"] as const;
export const DEFAULT_BET_STRATEGY: BetStrategy = "riskAdjusted";
export const LEAGUE_DEFAULT_BET_STRATEGY: Record<LeagueCode, BetStrategy> = {
  NHL: "riskAdjusted",
  NBA: "capitalPreservation",
  NCAAM: "riskAdjusted",
};
export type BetRiskRegime = "normal" | "guarded";

export type BetStrategyConfig = {
  label: string;
  shortLabel: string;
  description: string;
  allowUnderdogs: boolean;
  maxUnderdogMoneyline?: number | null;
  minEdge: number;
  minExpectedValue: number;
  stakeScale: number;
  maxBetBankrollPercent: number;
  maxDailyBankrollPercent: number | null;
};

export type BetStrategyRuleConfig = Pick<
  BetStrategyConfig,
  | "allowUnderdogs"
  | "maxUnderdogMoneyline"
  | "minEdge"
  | "minExpectedValue"
  | "stakeScale"
  | "maxBetBankrollPercent"
  | "maxDailyBankrollPercent"
>;

const SHARED_MIN_EDGE = 0.03;
const SHARED_MIN_EXPECTED_VALUE = 0.02;

type BetStrategyTuning = Omit<BetStrategyConfig, "description">;
type BetStrategyExperimentOverride = Partial<Pick<BetStrategyConfig, "stakeScale" | "maxBetBankrollPercent" | "maxDailyBankrollPercent">>;

const NBA_STATIC_TUNING_OVERRIDES: Partial<Record<BetStrategy, Partial<BetStrategyConfig>>> = {
  riskAdjusted: {
    minEdge: 0.05,
    minExpectedValue: 0.05,
    stakeScale: 0.25,
    maxBetBankrollPercent: 0.75,
    maxDailyBankrollPercent: 2.5,
    maxUnderdogMoneyline: 300,
  },
  aggressive: {
    minEdge: 0.05,
    minExpectedValue: 0.05,
    stakeScale: 0.4,
    maxBetBankrollPercent: 1,
    maxDailyBankrollPercent: 3,
    maxUnderdogMoneyline: 300,
  },
  capitalPreservation: {
    minEdge: 0.04,
    minExpectedValue: 0.03,
    maxBetBankrollPercent: 0.75,
    maxDailyBankrollPercent: 2.5,
  },
};

function formatPercent(value: number): string {
  return Number.isInteger(value) ? `${value}%` : `${value.toFixed(2)}%`;
}

function buildStrategyDescription(config: BetStrategyTuning): string {
  const dailyBudgetText =
    typeof config.maxDailyBankrollPercent === "number" && Number.isFinite(config.maxDailyBankrollPercent) && config.maxDailyBankrollPercent > 0
      ? `a ${formatPercent(config.maxDailyBankrollPercent)} nightly budget on the reference bankroll`
      : "no nightly budget cap";
  const longShotText =
    typeof config.maxUnderdogMoneyline === "number" && Number.isFinite(config.maxUnderdogMoneyline)
      ? ` Long-shot underdogs above +${Math.round(config.maxUnderdogMoneyline)} are skipped.`
      : "";
  return `${config.label} baseline with the shared value screen, a ${formatPercent(config.maxBetBankrollPercent)} per-bet cap, and ${dailyBudgetText}.${longShotText}`;
}

export const BETTING_STRATEGY_TUNING: Record<BetStrategy, BetStrategyTuning> = {
  riskAdjusted: {
    label: "Balanced",
    shortLabel: "Standard risk",
    allowUnderdogs: true,
    maxUnderdogMoneyline: null,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.5,
    maxBetBankrollPercent: 1.25,
    maxDailyBankrollPercent: 4,
  },
  aggressive: {
    label: "Aggressive",
    shortLabel: "Wider caps",
    allowUnderdogs: true,
    maxUnderdogMoneyline: null,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.75,
    maxBetBankrollPercent: 1.75,
    maxDailyBankrollPercent: 6,
  },
  capitalPreservation: {
    label: "Conservative",
    shortLabel: "Favorites only",
    allowUnderdogs: false,
    maxUnderdogMoneyline: null,
    minEdge: SHARED_MIN_EDGE,
    minExpectedValue: SHARED_MIN_EXPECTED_VALUE,
    stakeScale: 0.25,
    maxBetBankrollPercent: 0.75,
    maxDailyBankrollPercent: 2.5,
  },
};

const BET_STRATEGY_CONFIG: Record<BetStrategy, BetStrategyConfig> = Object.fromEntries(
  (Object.entries(BETTING_STRATEGY_TUNING) as [BetStrategy, BetStrategyTuning][]).map(([strategy, config]) => [
    strategy,
    {
      ...config,
      description: buildStrategyDescription(config),
    },
  ])
) as Record<BetStrategy, BetStrategyConfig>;

export const BETTING_STRATEGY_EXPERIMENT_OVERRIDES: Record<BetStrategy, BetStrategyExperimentOverride> = {
  riskAdjusted: {},
  aggressive: {},
  capitalPreservation: {},
};

function minNumber(left: number | null | undefined, right: number): number {
  if (typeof left !== "number" || !Number.isFinite(left)) return right;
  return Math.min(left, right);
}

function applyLeagueBetStrategyAdjustments(strategy: BetStrategy, config: BetStrategyConfig, league?: LeagueCode | null): BetStrategyConfig {
  if (league !== "NBA") {
    return config;
  }

  const overrides = NBA_STATIC_TUNING_OVERRIDES[strategy];
  if (!overrides) {
    return config;
  }

  const merged: BetStrategyTuning = {
    label: config.label,
    shortLabel: config.shortLabel,
    allowUnderdogs: overrides.allowUnderdogs ?? config.allowUnderdogs,
    maxUnderdogMoneyline:
      Object.prototype.hasOwnProperty.call(overrides, "maxUnderdogMoneyline")
        ? overrides.maxUnderdogMoneyline ?? null
        : config.maxUnderdogMoneyline ?? null,
    minEdge: overrides.minEdge ?? config.minEdge,
    minExpectedValue: overrides.minExpectedValue ?? config.minExpectedValue,
    stakeScale: overrides.stakeScale ?? config.stakeScale,
    maxBetBankrollPercent: overrides.maxBetBankrollPercent ?? config.maxBetBankrollPercent,
    maxDailyBankrollPercent:
      Object.prototype.hasOwnProperty.call(overrides, "maxDailyBankrollPercent")
        ? overrides.maxDailyBankrollPercent ?? null
        : config.maxDailyBankrollPercent,
  };

  return {
    ...config,
    ...merged,
    description: buildStrategyDescription(merged),
  };
}

function applyRiskRegimeAdjustments(
  strategy: BetStrategy,
  config: BetStrategyConfig,
  league?: LeagueCode | null,
  riskRegime: BetRiskRegime = "normal"
): BetStrategyConfig {
  if (league !== "NBA" || riskRegime !== "guarded") {
    return config;
  }

  const guarded: BetStrategyTuning = {
    label: config.label,
    shortLabel: config.shortLabel,
    allowUnderdogs: strategy === "capitalPreservation" ? false : false,
    maxUnderdogMoneyline: null,
    minEdge:
      strategy === "capitalPreservation"
        ? Math.max(config.minEdge, 0.05)
        : Math.max(config.minEdge, 0.06),
    minExpectedValue:
      strategy === "capitalPreservation"
        ? Math.max(config.minExpectedValue, 0.035)
        : Math.max(config.minExpectedValue, 0.05),
    stakeScale:
      strategy === "capitalPreservation"
        ? Math.min(config.stakeScale, 0.2)
        : Math.min(config.stakeScale, 0.2),
    maxBetBankrollPercent:
      strategy === "capitalPreservation"
        ? minNumber(config.maxBetBankrollPercent, 0.5)
        : minNumber(config.maxBetBankrollPercent, 0.5),
    maxDailyBankrollPercent:
      strategy === "capitalPreservation"
        ? minNumber(config.maxDailyBankrollPercent, 1.5)
        : minNumber(config.maxDailyBankrollPercent, 1.5),
  };

  return {
    ...config,
    ...guarded,
    description: buildStrategyDescription(guarded),
  };
}

export function applyBetStrategyExperimentOverrides(strategy: BetStrategy, config: BetStrategyConfig): BetStrategyConfig {
  const overrides = BETTING_STRATEGY_EXPERIMENT_OVERRIDES[strategy];
  if (!overrides || Object.keys(overrides).length === 0) {
    return config;
  }

  const merged: BetStrategyTuning = {
    label: config.label,
    shortLabel: config.shortLabel,
    allowUnderdogs: config.allowUnderdogs,
    maxUnderdogMoneyline: config.maxUnderdogMoneyline ?? null,
    minEdge: config.minEdge,
    minExpectedValue: config.minExpectedValue,
    stakeScale: overrides.stakeScale ?? config.stakeScale,
    maxBetBankrollPercent: overrides.maxBetBankrollPercent ?? config.maxBetBankrollPercent,
    maxDailyBankrollPercent:
      Object.prototype.hasOwnProperty.call(overrides, "maxDailyBankrollPercent")
        ? overrides.maxDailyBankrollPercent ?? null
        : config.maxDailyBankrollPercent,
  };

  return {
    ...config,
    ...merged,
    description: buildStrategyDescription(merged),
  };
}

export function normalizeBetStrategy(value?: string | null): BetStrategy {
  const normalized = String(value || "").trim().toLowerCase();
  switch (normalized) {
    case "riskadjusted":
    case "risk-adjusted":
    case "risk_adjusted":
    case "optimal":
    case "sharpe":
    case "sharpe-like":
    case "balanced":
      return "riskAdjusted";
    case "aggressiveev":
    case "aggressive-ev":
    case "aggressive_ev":
    case "highev":
    case "high-ev":
    case "high_ev":
    case "aggressive":
    case "riskloving":
    case "risk-loving":
    case "risk_loving":
      return "aggressive";
    case "capitalpreservation":
    case "capital-preservation":
    case "capital_preservation":
    case "preservation":
    case "lowrisk":
    case "low-risk":
    case "low_risk":
    case "riskaverse":
    case "risk-averse":
    case "risk_averse":
    case "conservative":
      return "capitalPreservation";
    default:
      return DEFAULT_BET_STRATEGY;
  }
}

export function getDefaultBetStrategyForLeague(league?: LeagueCode | null): BetStrategy {
  if (!league) {
    return DEFAULT_BET_STRATEGY;
  }
  return LEAGUE_DEFAULT_BET_STRATEGY[league] || DEFAULT_BET_STRATEGY;
}

export function strategyFromRequest(request: Request): BetStrategy {
  const url = new URL(request.url);
  const leagueParam = url.searchParams.get("league");
  const league =
    leagueParam === "NHL" || leagueParam === "NBA" || leagueParam === "NCAAM" ? leagueParam : null;
  const strategyParam = url.searchParams.get("strategy");
  return strategyParam ? normalizeBetStrategy(strategyParam) : getDefaultBetStrategyForLeague(league);
}

export function getBetStrategyConfig(
  strategy: BetStrategy,
  options?: {
    league?: LeagueCode | null;
    riskRegime?: BetRiskRegime;
  }
): BetStrategyConfig {
  const experimented = applyBetStrategyExperimentOverrides(strategy, BET_STRATEGY_CONFIG[strategy]);
  const leagueAdjusted = applyLeagueBetStrategyAdjustments(strategy, experimented, options?.league);
  return applyRiskRegimeAdjustments(strategy, leagueAdjusted, options?.league, options?.riskRegime);
}

export function toBetStrategyRuleConfig(strategyConfig: BetStrategyConfig): BetStrategyRuleConfig {
  return {
    allowUnderdogs: strategyConfig.allowUnderdogs,
    maxUnderdogMoneyline: strategyConfig.maxUnderdogMoneyline ?? null,
    minEdge: strategyConfig.minEdge,
    minExpectedValue: strategyConfig.minExpectedValue,
    stakeScale: strategyConfig.stakeScale,
    maxBetBankrollPercent: strategyConfig.maxBetBankrollPercent,
    maxDailyBankrollPercent: strategyConfig.maxDailyBankrollPercent,
  };
}

export function getBetStrategyLabel(strategy: BetStrategy): string {
  return BET_STRATEGY_CONFIG[strategy].label;
}
