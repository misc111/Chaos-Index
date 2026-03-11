import {
  DEFAULT_BET_STRATEGY,
  getBetStrategyConfig,
  type BetStrategyRuleConfig,
  type BetStrategy,
} from "@/lib/betting-strategy";
import type { ModelWinProbabilities } from "@/lib/betting-model";

export type ExpectedSide = "home" | "away" | "none";

export type BetInput = {
  home_team: string;
  away_team: string;
  home_win_probability: number;
  home_moneyline?: number | null;
  away_moneyline?: number | null;
  betting_model_name?: string | null;
  model_win_probabilities?: ModelWinProbabilities | null;
};

export type BetDecision = {
  bet: string;
  reason: string;
  side: ExpectedSide;
  team: string | null;
  stake: number;
  odds: number | null;
  modelProbability: number | null;
  marketProbability: number | null;
  edge: number | null;
  expectedValue: number | null;
};

export type BetSettlement = {
  outcome: "win" | "loss" | "no_bet";
  profit: number;
  payout: number;
};

export type BetDecisionTrace = {
  decision: BetDecision;
  strategyLabel: string;
  strategyConfig: BetStrategyRuleConfig;
  homeRawModelProbability: number | null;
  awayRawModelProbability: number | null;
  homeAdjustedProbability: number | null;
  awayAdjustedProbability: number | null;
  homeReferenceProbability: number | null;
  awayReferenceProbability: number | null;
  homeFairProbability: number | null;
  awayFairProbability: number | null;
  homeExpectedValue: number | null;
  awayExpectedValue: number | null;
  candidateSide: ExpectedSide;
  candidateTeam: string | null;
  candidateOdds: number | null;
  candidateIsUnderdog: boolean | null;
  candidateRawModelProbability: number | null;
  candidateAdjustedProbability: number | null;
  candidateReferenceProbability: number | null;
  candidateMarketProbability: number | null;
  candidateEdge: number | null;
  candidateExpectedValue: number | null;
  candidateConfidenceWeight: number | null;
  baseStakeShareOfBankroll: number | null;
  scaledStakeShareOfBankroll: number | null;
  cappedStakeShareOfBankroll: number | null;
  quotedStake: number;
  finalStake: number;
  peerConsensusProbability?: number | null;
  consensusGap?: number | null;
  preDailyCapStake?: number | null;
  dailyRiskCapApplied?: boolean;
  gates: {
    oddsAvailable: boolean;
    positiveExpectedValue: boolean;
    edge: boolean;
    expectedValue: boolean;
    underdogAllowed: boolean;
    dailyBudget: boolean;
  };
};

// Single source of truth for the bankroll assumptions used across sizing,
// replay materialization, and UI copy.
export const REFERENCE_BANKROLL_DOLLARS = 5_000;
export const REFERENCE_STAKE_BANKROLL_FRACTION = 0.01;
export const REFERENCE_STAKE_DOLLARS = Math.round(REFERENCE_BANKROLL_DOLLARS * REFERENCE_STAKE_BANKROLL_FRACTION);
export const HISTORICAL_BANKROLL_START_DOLLARS = REFERENCE_BANKROLL_DOLLARS;
export const HISTORICAL_BANKROLL_START_DATE_CENTRAL = "2026-03-05";

const STAKE_ROUNDING_DOLLARS = 5;
const REFERENCE_MARKET_WEIGHT = 0.7;
const REFERENCE_PEER_WEIGHT = 0.3;
const MIN_MODEL_CONFIDENCE_WEIGHT = 0.25;
const FULL_MARGIN_FOR_FULL_WEIGHT = 0.2;
const MIN_PEER_AGREEMENT_WEIGHT = 0.55;
const PEER_DISAGREEMENT_FOR_MIN_WEIGHT = 0.2;

type BetDisplayRecommendation = {
  team: string | null;
  stake: number;
  reason: string;
};

type ProbabilityAdjustment = {
  referenceProbability: number;
  adjustedProbability: number;
  confidenceWeight: number;
  peerConsensusProbability: number | null;
  consensusGap: number | null;
};

type TraceContext = {
  strategyLabel: string;
  strategyConfig: BetStrategyRuleConfig;
};

function dollarsFromBankrollShare(share: number): number {
  return REFERENCE_BANKROLL_DOLLARS * share;
}

function roundStakeAmount(amount: number): number {
  if (!Number.isFinite(amount) || amount <= 0) return 0;
  const rounded = Math.round(amount / STAKE_ROUNDING_DOLLARS) * STAKE_ROUNDING_DOLLARS;
  return rounded >= STAKE_ROUNDING_DOLLARS ? rounded : 0;
}

function clampProbability(value: number): number {
  if (!Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, value));
}

function clampPositive(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, value);
}

function decimalOddsToBaseStakeShare(probability: number, decimalOdds: number): number | null {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) return null;
  if (!Number.isFinite(decimalOdds) || decimalOdds <= 1) return null;

  const netOdds = decimalOdds - 1;
  const fraction = (probability * decimalOdds - 1) / netOdds;
  return Number.isFinite(fraction) ? fraction : null;
}

function quotedStakeFromBaseShare(baseStakeShare: number, stakeScale: number, maxBetBankrollPercent: number): number {
  if (!Number.isFinite(baseStakeShare) || baseStakeShare <= 0) return 0;
  if (!Number.isFinite(stakeScale) || stakeScale <= 0) return 0;

  const scaledShare = clampPositive(baseStakeShare * stakeScale);
  const cappedShare = Math.min(maxBetBankrollPercent / 100, scaledShare);
  return roundStakeAmount(dollarsFromBankrollShare(cappedShare));
}

function sideProbabilityFromHomeProbability(homeProbability: number, side: ExpectedSide): number {
  if (side === "home") return clampProbability(homeProbability);
  if (side === "away") return clampProbability(1 - homeProbability);
  return 0.5;
}

function resolvePeerConsensusProbability(
  row: BetInput,
  side: ExpectedSide
): { peerConsensusProbability: number | null; consensusGap: number | null } {
  if (side === "none") {
    return {
      peerConsensusProbability: null,
      consensusGap: null,
    };
  }

  const bettingModelName = String(row.betting_model_name || "").trim();
  const modelProbabilities = row.model_win_probabilities || {};
  if (!bettingModelName) {
    return {
      peerConsensusProbability: null,
      consensusGap: null,
    };
  }

  const selectedHomeProbability =
    typeof modelProbabilities[bettingModelName] === "number"
      ? clampProbability(Number(modelProbabilities[bettingModelName]))
      : clampProbability(Number(row.home_win_probability));

  const peerSideProbabilities = Object.entries(modelProbabilities)
    .filter(([modelName, probability]) => modelName !== bettingModelName && typeof probability === "number" && Number.isFinite(probability))
    .map(([, probability]) => sideProbabilityFromHomeProbability(Number(probability), side));

  if (!peerSideProbabilities.length) {
    return {
      peerConsensusProbability: null,
      consensusGap: null,
    };
  }

  const peerConsensusProbability = peerSideProbabilities.reduce((sum, probability) => sum + probability, 0) / peerSideProbabilities.length;
  const consensusGap = sideProbabilityFromHomeProbability(selectedHomeProbability, side) - peerConsensusProbability;
  return {
    peerConsensusProbability,
    consensusGap,
  };
}

function buildProbabilityAdjustment(
  rawProbability: number,
  fairProbability: number,
  peerConsensusProbability: number | null
): ProbabilityAdjustment {
  const normalizedRaw = clampProbability(rawProbability);
  const normalizedFair = clampProbability(fairProbability);
  const normalizedPeer =
    typeof peerConsensusProbability === "number" && Number.isFinite(peerConsensusProbability)
      ? clampProbability(peerConsensusProbability)
      : null;

  const referenceProbability =
    normalizedPeer === null
      ? normalizedFair
      : clampProbability(normalizedFair * REFERENCE_MARKET_WEIGHT + normalizedPeer * REFERENCE_PEER_WEIGHT);

  const margin = Math.abs(normalizedRaw - 0.5);
  const marginWeight =
    margin >= FULL_MARGIN_FOR_FULL_WEIGHT
      ? 1
      : MIN_MODEL_CONFIDENCE_WEIGHT +
        ((1 - MIN_MODEL_CONFIDENCE_WEIGHT) * margin) / FULL_MARGIN_FOR_FULL_WEIGHT;

  const peerAgreementWeight =
    normalizedPeer === null
      ? 1
      : Math.max(
          MIN_PEER_AGREEMENT_WEIGHT,
          1 - Math.abs(normalizedRaw - normalizedPeer) / PEER_DISAGREEMENT_FOR_MIN_WEIGHT
        );

  const confidenceWeight = Math.max(
    MIN_MODEL_CONFIDENCE_WEIGHT,
    Math.min(1, marginWeight * peerAgreementWeight)
  );
  const adjustedProbability = clampProbability(
    referenceProbability + confidenceWeight * (normalizedRaw - referenceProbability)
  );

  return {
    referenceProbability,
    adjustedProbability,
    confidenceWeight,
    peerConsensusProbability: normalizedPeer,
    consensusGap: normalizedPeer === null ? null : normalizedRaw - normalizedPeer,
  };
}

function buildPricedBetReason(candidateIsUnderdog: boolean): string {
  return candidateIsUnderdog ? "Underdog underpriced after uncertainty adjustment" : "Favorite underpriced after uncertainty adjustment";
}

function formatStakeForDecision(team: string | null, stake: number): string {
  if (stake <= 0 || !team) return "$0";
  if (!Number.isFinite(stake)) return "$0";
  const fractionDigits = Number.isInteger(stake) ? 0 : 2;
  return `$${stake.toFixed(fractionDigits)} ${team}`;
}

function buildDecision(
  row: BetInput,
  side: ExpectedSide,
  stake: number,
  reason: string,
  adjustedProb?: number | null,
  fairProb?: number | null,
  ev?: number | null,
  edge?: number | null
): BetDecision {
  const team = side === "home" ? row.home_team : side === "away" ? row.away_team : null;
  const odds = side === "home" ? Number(row.home_moneyline) : side === "away" ? Number(row.away_moneyline) : null;

  return {
    bet: formatStakeForDecision(team, stake),
    reason,
    side,
    team,
    stake,
    odds: Number.isFinite(odds) ? odds : null,
    modelProbability: typeof adjustedProb === "number" && Number.isFinite(adjustedProb) ? adjustedProb : null,
    marketProbability: typeof fairProb === "number" && Number.isFinite(fairProb) ? fairProb : null,
    edge: typeof edge === "number" && Number.isFinite(edge) ? edge : null,
    expectedValue: typeof ev === "number" && Number.isFinite(ev) ? ev : null,
  };
}

function buildDefaultTraceFields(): Omit<
  BetDecisionTrace,
  "decision" | "strategyLabel" | "strategyConfig" | "gates"
> {
  return {
    homeRawModelProbability: null,
    awayRawModelProbability: null,
    homeAdjustedProbability: null,
    awayAdjustedProbability: null,
    homeReferenceProbability: null,
    awayReferenceProbability: null,
    homeFairProbability: null,
    awayFairProbability: null,
    homeExpectedValue: null,
    awayExpectedValue: null,
    candidateSide: "none",
    candidateTeam: null,
    candidateOdds: null,
    candidateIsUnderdog: null,
    candidateRawModelProbability: null,
    candidateAdjustedProbability: null,
    candidateReferenceProbability: null,
    candidateMarketProbability: null,
    candidateEdge: null,
    candidateExpectedValue: null,
    candidateConfidenceWeight: null,
    baseStakeShareOfBankroll: null,
    scaledStakeShareOfBankroll: null,
    cappedStakeShareOfBankroll: null,
    quotedStake: 0,
    finalStake: 0,
    peerConsensusProbability: null,
    consensusGap: null,
    preDailyCapStake: null,
    dailyRiskCapApplied: false,
  };
}

function buildTrace(
  context: TraceContext,
  decision: BetDecision,
  gates: BetDecisionTrace["gates"],
  overrides: Partial<Omit<BetDecisionTrace, "decision" | "strategyLabel" | "strategyConfig" | "gates">> = {}
): BetDecisionTrace {
  return {
    ...buildDefaultTraceFields(),
    ...overrides,
    decision,
    strategyLabel: context.strategyLabel,
    strategyConfig: context.strategyConfig,
    gates,
  };
}

function applyStakeOverride(trace: BetDecisionTrace, stake: number, reason: string, dailyRiskCapApplied: boolean): BetDecisionTrace {
  const adjustedStake = roundStakeAmount(stake);
  const team = trace.decision.team;
  const nextDecision = {
    ...trace.decision,
    stake: adjustedStake,
    bet: formatStakeForDecision(team, adjustedStake),
    reason,
  };

  return {
    ...trace,
    decision: nextDecision,
    finalStake: adjustedStake,
    preDailyCapStake: trace.quotedStake,
    dailyRiskCapApplied,
    gates: {
      ...trace.gates,
      dailyBudget: adjustedStake > 0,
    },
  };
}

function capStakeToRemainingBudget(trace: BetDecisionTrace, remainingBudget: number): number {
  if (remainingBudget <= 0) return 0;
  return roundStakeAmount(Math.min(trace.finalStake, remainingBudget));
}

function compareByExpectedValue(left: BetDecisionTrace, right: BetDecisionTrace): number {
  return (
    (right.candidateExpectedValue ?? Number.NEGATIVE_INFINITY) - (left.candidateExpectedValue ?? Number.NEGATIVE_INFINITY) ||
    (right.candidateEdge ?? Number.NEGATIVE_INFINITY) - (left.candidateEdge ?? Number.NEGATIVE_INFINITY) ||
    (right.finalStake ?? 0) - (left.finalStake ?? 0) ||
    String(left.decision.team || "").localeCompare(String(right.decision.team || ""))
  );
}

function resolveDailyRiskBudgetDollars(
  strategyConfig?: Pick<BetStrategyRuleConfig, "maxDailyBankrollPercent"> | null
): number | null {
  const maxDailyBankrollPercent = strategyConfig?.maxDailyBankrollPercent;
  if (
    typeof maxDailyBankrollPercent !== "number" ||
    !Number.isFinite(maxDailyBankrollPercent) ||
    maxDailyBankrollPercent <= 0
  ) {
    return null;
  }

  return dollarsFromBankrollShare(maxDailyBankrollPercent / 100);
}

export function applyDailyRiskCapToDecisionTraces(traces: BetDecisionTrace[]): BetDecisionTrace[] {
  if (!traces.length) return traces;

  const strategyConfig = traces[0]?.strategyConfig;
  const budgetDollars = resolveDailyRiskBudgetDollars(strategyConfig);
  if (budgetDollars === null) {
    return traces.map((trace) => ({
      ...trace,
      gates: {
        ...trace.gates,
        dailyBudget: trace.finalStake > 0,
      },
    }));
  }

  let usedDollars = 0;
  const next = [...traces];

  const ranked = traces
    .map((trace, index) => ({ trace, index }))
    .filter(({ trace }) => trace.finalStake > 0)
    .sort((left, right) => compareByExpectedValue(left.trace, right.trace));

  for (const { index } of ranked) {
    const trace = next[index];
    const remainingBudget = budgetDollars - usedDollars;
    const cappedStake = capStakeToRemainingBudget(trace, remainingBudget);

    if (cappedStake <= 0) {
      next[index] = applyStakeOverride(trace, 0, "Daily risk budget exhausted", true);
      continue;
    }

    if (cappedStake < trace.finalStake) {
      next[index] = applyStakeOverride(trace, cappedStake, `${trace.decision.reason}; daily risk cap`, true);
    } else {
      next[index] = {
        ...trace,
        gates: {
          ...trace.gates,
          dailyBudget: true,
        },
      };
    }

    usedDollars += next[index].finalStake;
  }

  return next.map((trace) =>
    trace.finalStake > 0
      ? trace
      : {
          ...trace,
          gates: {
            ...trace.gates,
            dailyBudget: trace.gates.dailyBudget,
          },
        }
  );
}

export function explainBetDecision(
  row: BetInput,
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  strategyConfigOverride?: BetStrategyRuleConfig,
  strategyLabelOverride?: string
): BetDecisionTrace {
  const strategyConfig = strategyConfigOverride || getBetStrategyConfig(strategy);
  const strategyLabel = strategyLabelOverride || getBetStrategyConfig(strategy).label;
  const context: TraceContext = { strategyLabel, strategyConfig };
  const homeOdds = Number(row.home_moneyline);
  const awayOdds = Number(row.away_moneyline);
  const oddsAvailable = Number.isFinite(homeOdds) && Number.isFinite(awayOdds) && homeOdds !== 0 && awayOdds !== 0;

  const baseGates: BetDecisionTrace["gates"] = {
    oddsAvailable,
    positiveExpectedValue: false,
    edge: false,
    expectedValue: false,
    underdogAllowed: false,
    dailyBudget: true,
  };

  if (!oddsAvailable) {
    return buildTrace(context, buildDecision(row, "none", 0, "Missing odds"), baseGates);
  }

  const pHomeRaw = clampProbability(Number(row.home_win_probability));
  const pAwayRaw = 1 - pHomeRaw;
  const impHome = americanToImpliedProbability(homeOdds);
  const impAway = americanToImpliedProbability(awayOdds);
  if (impHome === null || impAway === null) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, "Missing odds"),
      baseGates,
      {
        homeRawModelProbability: pHomeRaw,
        awayRawModelProbability: pAwayRaw,
      }
    );
  }

  const impTotal = impHome + impAway;
  if (!Number.isFinite(impTotal) || impTotal <= 0) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, "Missing odds"),
      baseGates,
      {
        homeRawModelProbability: pHomeRaw,
        awayRawModelProbability: pAwayRaw,
      }
    );
  }

  const fairHome = impHome / impTotal;
  const fairAway = impAway / impTotal;
  const homeConsensus = resolvePeerConsensusProbability(row, "home");
  const homeAdjustment = buildProbabilityAdjustment(pHomeRaw, fairHome, homeConsensus.peerConsensusProbability);
  const awayAdjustment = {
    referenceProbability: clampProbability(1 - homeAdjustment.referenceProbability),
    adjustedProbability: clampProbability(1 - homeAdjustment.adjustedProbability),
    confidenceWeight: homeAdjustment.confidenceWeight,
    peerConsensusProbability:
      typeof homeAdjustment.peerConsensusProbability === "number"
        ? clampProbability(1 - homeAdjustment.peerConsensusProbability)
        : null,
    consensusGap: homeConsensus.consensusGap === null ? null : -homeConsensus.consensusGap,
  };

  const decHome = americanToDecimalOdds(homeOdds);
  const decAway = americanToDecimalOdds(awayOdds);
  if (decHome === null || decAway === null) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, "Missing odds"),
      baseGates,
      {
        homeRawModelProbability: pHomeRaw,
        awayRawModelProbability: pAwayRaw,
        homeAdjustedProbability: homeAdjustment.adjustedProbability,
        awayAdjustedProbability: awayAdjustment.adjustedProbability,
        homeReferenceProbability: homeAdjustment.referenceProbability,
        awayReferenceProbability: awayAdjustment.referenceProbability,
        homeFairProbability: fairHome,
        awayFairProbability: fairAway,
        peerConsensusProbability: homeConsensus.peerConsensusProbability,
        consensusGap: homeConsensus.consensusGap,
      }
    );
  }

  const evHome = homeAdjustment.adjustedProbability * decHome - 1;
  const evAway = awayAdjustment.adjustedProbability * decAway - 1;
  const positiveExpectedValue = evHome > 0 || evAway > 0;
  if (!positiveExpectedValue) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, "Adjusted price fair"),
      {
        ...baseGates,
        positiveExpectedValue,
      },
      {
        homeRawModelProbability: pHomeRaw,
        awayRawModelProbability: pAwayRaw,
        homeAdjustedProbability: homeAdjustment.adjustedProbability,
        awayAdjustedProbability: awayAdjustment.adjustedProbability,
        homeReferenceProbability: homeAdjustment.referenceProbability,
        awayReferenceProbability: awayAdjustment.referenceProbability,
        homeFairProbability: fairHome,
        awayFairProbability: fairAway,
        homeExpectedValue: evHome,
        awayExpectedValue: evAway,
        peerConsensusProbability: homeConsensus.peerConsensusProbability,
        consensusGap: homeConsensus.consensusGap,
      }
    );
  }

  const side = evHome > evAway ? "home" : evAway > evHome ? "away" : homeAdjustment.adjustedProbability >= awayAdjustment.adjustedProbability ? "home" : "away";
  const rawModelProb = side === "home" ? pHomeRaw : pAwayRaw;
  const adjustedProb = side === "home" ? homeAdjustment.adjustedProbability : awayAdjustment.adjustedProbability;
  const referenceProb = side === "home" ? homeAdjustment.referenceProbability : awayAdjustment.referenceProbability;
  const fairProb = side === "home" ? fairHome : fairAway;
  const edge = adjustedProb - fairProb;
  const ev = side === "home" ? evHome : evAway;
  const candidateTeam = side === "home" ? row.home_team : row.away_team;
  const candidateOdds = side === "home" ? homeOdds : awayOdds;
  const candidateIsUnderdog = candidateOdds > 0;
  const edgeGate = edge >= strategyConfig.minEdge;
  const expectedValueGate = ev >= strategyConfig.minExpectedValue;
  const underdogAllowed = strategyConfig.allowUnderdogs || !candidateIsUnderdog;

  const traceOverrides: Partial<Omit<BetDecisionTrace, "decision" | "strategyLabel" | "strategyConfig" | "gates">> = {
    homeRawModelProbability: pHomeRaw,
    awayRawModelProbability: pAwayRaw,
    homeAdjustedProbability: homeAdjustment.adjustedProbability,
    awayAdjustedProbability: awayAdjustment.adjustedProbability,
    homeReferenceProbability: homeAdjustment.referenceProbability,
    awayReferenceProbability: awayAdjustment.referenceProbability,
    homeFairProbability: fairHome,
    awayFairProbability: fairAway,
    homeExpectedValue: evHome,
    awayExpectedValue: evAway,
    candidateSide: side,
    candidateTeam,
    candidateOdds,
    candidateIsUnderdog,
    candidateRawModelProbability: rawModelProb,
    candidateAdjustedProbability: adjustedProb,
    candidateReferenceProbability: referenceProb,
    candidateMarketProbability: fairProb,
    candidateEdge: edge,
    candidateExpectedValue: ev,
    candidateConfidenceWeight: side === "home" ? homeAdjustment.confidenceWeight : awayAdjustment.confidenceWeight,
    peerConsensusProbability: side === "home" ? homeConsensus.peerConsensusProbability : awayAdjustment.peerConsensusProbability,
    consensusGap: side === "home" ? homeConsensus.consensusGap : awayAdjustment.consensusGap,
  };

  if (!edgeGate || !expectedValueGate) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, "Adjusted price fair"),
      {
        ...baseGates,
        positiveExpectedValue,
        edge: edgeGate,
        expectedValue: expectedValueGate,
        underdogAllowed,
      },
      traceOverrides
    );
  }

  if (!underdogAllowed) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, `${strategyLabel} skips underdogs`, adjustedProb, fairProb, ev, edge),
      {
        ...baseGates,
        positiveExpectedValue,
        edge: edgeGate,
        expectedValue: expectedValueGate,
        underdogAllowed,
      },
      traceOverrides
    );
  }

  const sideDecimalOdds = side === "home" ? decHome : decAway;
  const baseStakeShareOfBankroll = decimalOddsToBaseStakeShare(adjustedProb, sideDecimalOdds);
  const scaledStakeShareOfBankroll =
    baseStakeShareOfBankroll === null ? null : clampPositive(strategyConfig.stakeScale * baseStakeShareOfBankroll);
  const cappedStakeShareOfBankroll =
    scaledStakeShareOfBankroll === null
      ? null
      : Math.min(strategyConfig.maxBetBankrollPercent / 100, scaledStakeShareOfBankroll);
  const quotedStake =
    baseStakeShareOfBankroll === null
      ? 0
      : quotedStakeFromBaseShare(baseStakeShareOfBankroll, strategyConfig.stakeScale, strategyConfig.maxBetBankrollPercent);
  const stake = quotedStake;

  if (stake <= 0) {
    return buildTrace(
      context,
      buildDecision(row, "none", 0, "Adjusted price fair"),
      {
        ...baseGates,
        positiveExpectedValue,
        edge: edgeGate,
        expectedValue: expectedValueGate,
        underdogAllowed,
      },
      {
        ...traceOverrides,
        baseStakeShareOfBankroll,
        scaledStakeShareOfBankroll,
        cappedStakeShareOfBankroll,
        quotedStake,
      }
    );
  }

  const decision = buildDecision(
    row,
    side,
    stake,
    buildPricedBetReason(candidateIsUnderdog),
    adjustedProb,
    fairProb,
    ev,
    edge
  );

  return buildTrace(
    context,
    decision,
    {
      ...baseGates,
      positiveExpectedValue,
      edge: edgeGate,
      expectedValue: expectedValueGate,
      underdogAllowed,
    },
    {
      ...traceOverrides,
      baseStakeShareOfBankroll,
      scaledStakeShareOfBankroll,
      cappedStakeShareOfBankroll,
      quotedStake,
      finalStake: stake,
    }
  );
}

export function explainBetDecisionsForSlate(
  rows: BetInput[],
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  strategyConfigOverride?: BetStrategyRuleConfig,
  strategyLabelOverride?: string
): BetDecisionTrace[] {
  return applyDailyRiskCapToDecisionTraces(
    rows.map((row) => explainBetDecision(row, strategy, strategyConfigOverride, strategyLabelOverride))
  );
}

export function computeBetDecision(
  row: BetInput,
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  strategyConfigOverride?: BetStrategyRuleConfig
): BetDecision {
  return explainBetDecision(row, strategy, strategyConfigOverride).decision;
}

export function computeBetDecisionsForSlate(
  rows: BetInput[],
  strategy: BetStrategy = DEFAULT_BET_STRATEGY,
  strategyConfigOverride?: BetStrategyRuleConfig,
  strategyLabelOverride?: string
): BetDecision[] {
  return explainBetDecisionsForSlate(rows, strategy, strategyConfigOverride, strategyLabelOverride).map(
    (trace) => trace.decision
  );
}

export function expectedSide(homeWinProbability: number): ExpectedSide {
  if (homeWinProbability > 0.5) return "home";
  if (homeWinProbability < 0.5) return "away";
  return "none";
}

export function expectedWinChance(homeWinProbability: number, side: ExpectedSide): number {
  if (side === "home") return clampProbability(homeWinProbability);
  if (side === "away") return clampProbability(1 - homeWinProbability);
  return 0.5;
}

export function americanToImpliedProbability(odds: number): number | null {
  if (!Number.isFinite(odds) || odds === 0) return null;
  if (odds > 0) return 100 / (odds + 100);
  const absOdds = Math.abs(odds);
  return absOdds / (absOdds + 100);
}

export function americanToDecimalOdds(odds: number): number | null {
  if (!Number.isFinite(odds) || odds === 0) return null;
  if (odds > 0) return 1 + odds / 100;
  return 1 + 100 / Math.abs(odds);
}

export function formatBetLabel(team: string | null, stake: number): string {
  return formatStakeForDecision(team, stake);
}

export function formatBetRecommendationLabel(team: string | null, stake: number): string {
  return formatBetLabel(team, stake);
}

export function formatBetRecommendation(recommendation: BetDisplayRecommendation): { label: string; reason: string } {
  return {
    label: formatBetRecommendationLabel(recommendation.team, recommendation.stake),
    reason: recommendation.reason,
  };
}

export function settleBet(decision: BetDecision, outcomeHomeWin?: number | boolean | null): BetSettlement {
  const homeWin = typeof outcomeHomeWin === "boolean" ? outcomeHomeWin : Number(outcomeHomeWin) === 1;
  if (decision.stake <= 0 || decision.side === "none" || !Number.isFinite(decision.odds)) {
    return { outcome: "no_bet", profit: 0, payout: 0 };
  }

  const won = (decision.side === "home" && homeWin) || (decision.side === "away" && !homeWin);
  if (!won) {
    return { outcome: "loss", profit: -decision.stake, payout: 0 };
  }

  const odds = Number(decision.odds);
  const profit = odds > 0 ? decision.stake * (odds / 100) : decision.stake * (100 / Math.abs(odds));
  return { outcome: "win", profit, payout: decision.stake + profit };
}
