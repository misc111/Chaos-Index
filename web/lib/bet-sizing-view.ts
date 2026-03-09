import { explainBetDecisionsForSlate, type BetDecisionTrace } from "@/lib/betting";
import type { BetStrategyPerformanceSnapshot, FrontierPointSummary, ResolvedBetStrategyConfig } from "@/lib/betting-optimizer";
import { dateKeyForScheduledGame } from "@/lib/games-today";
import { formatCentralDateLabel, formatCentralDateSummary } from "@/lib/games-today";
import type { BetStrategy } from "@/lib/betting-strategy";
import type { GamesTodayResponse, GamesTodayRow } from "@/lib/types";

export type BetSizingPolicyPreview = {
  key: string;
  label: string;
  shortLabel: string;
  description: string;
  matchingStrategies: BetStrategy[];
  configSignature: string;
  allowUnderdogs: boolean;
  minEdge: number;
  minExpectedValue: number;
  stakeScale: number;
  maxBetBankrollPercent: number;
  maxDailyBankrollPercent: number;
  optimizationSource: "historical_frontier" | "historical_downside" | "static_fallback" | "frontier_preview";
  metrics: BetStrategyPerformanceSnapshot | null;
  frontierPoint: FrontierPointSummary | null;
  isFrontierPoint: boolean;
};

export type BetSizingSlate = {
  label: string;
  rows: GamesTodayRow[];
  source: "upcoming" | "historical";
};

export type BetSizingGamePreview = {
  row: GamesTodayRow;
  trace: BetDecisionTrace;
};

function appendMatchingStrategy(existing: BetSizingPolicyPreview, strategy: BetStrategy): BetSizingPolicyPreview {
  if (existing.matchingStrategies.includes(strategy)) {
    return existing;
  }

  return {
    ...existing,
    matchingStrategies: [...existing.matchingStrategies, strategy],
  };
}

function buildPolicyFromStrategy(strategy: BetStrategy, config: ResolvedBetStrategyConfig): BetSizingPolicyPreview {
  return {
    key: strategy,
    label: config.label,
    shortLabel: config.shortLabel,
    description: config.description,
    matchingStrategies: [strategy],
    configSignature: config.config_signature,
    allowUnderdogs: config.allowUnderdogs,
    minEdge: config.minEdge,
    minExpectedValue: config.minExpectedValue,
    stakeScale: config.stakeScale,
    maxBetBankrollPercent: config.maxBetBankrollPercent,
    maxDailyBankrollPercent: config.maxDailyBankrollPercent,
    optimizationSource: config.optimization_source,
    metrics: config.metrics,
    frontierPoint: null,
    isFrontierPoint: false,
  };
}

function buildPolicyFromFrontierPoint(point: FrontierPointSummary): BetSizingPolicyPreview {
  return {
    key: point.config_signature,
    label: point.allowUnderdogs ? "Replay Preview: Dogs Allowed" : "Replay Preview: Favorites Only",
    shortLabel: `${point.maxBetBankrollPercent.toFixed(2)}% max bet · ${point.maxDailyBankrollPercent.toFixed(1)}% nightly`,
    description: "Preview a different replay-tested policy without changing the saved defaults.",
    matchingStrategies: [],
    configSignature: point.config_signature,
    allowUnderdogs: point.allowUnderdogs,
    minEdge: point.minEdge,
    minExpectedValue: point.minExpectedValue,
    stakeScale: point.stakeScale,
    maxBetBankrollPercent: point.maxBetBankrollPercent,
    maxDailyBankrollPercent: point.maxDailyBankrollPercent,
    optimizationSource: "frontier_preview",
    metrics: point,
    frontierPoint: point,
    isFrontierPoint: true,
  };
}

export function collectBetSizingPolicies(
  strategyConfigs: Record<BetStrategy, ResolvedBetStrategyConfig>,
  frontier: FrontierPointSummary[]
): {
  policies: BetSizingPolicyPreview[];
  frontierPolicies: BetSizingPolicyPreview[];
  byKey: Map<string, BetSizingPolicyPreview>;
} {
  const byKey = new Map<string, BetSizingPolicyPreview>();
  const orderedKeys: string[] = [];

  for (const strategy of Object.keys(strategyConfigs) as BetStrategy[]) {
    const config = strategyConfigs[strategy];
    const existing = byKey.get(config.config_signature);
    if (existing) {
      byKey.set(config.config_signature, appendMatchingStrategy(existing, strategy));
      continue;
    }

    const next = buildPolicyFromStrategy(strategy, config);
    byKey.set(next.configSignature, next);
    orderedKeys.push(next.configSignature);
  }

  const frontierPolicies = frontier.map((point) => {
    const existing = byKey.get(point.config_signature);
    if (existing) {
      const merged: BetSizingPolicyPreview = {
        ...existing,
        metrics: existing.metrics || point,
        frontierPoint: point,
        isFrontierPoint: true,
      };
      byKey.set(point.config_signature, merged);
      return merged;
    }

    const next = buildPolicyFromFrontierPoint(point);
    byKey.set(next.configSignature, next);
    orderedKeys.push(next.configSignature);
    return next;
  });

  return {
    policies: orderedKeys.map((key) => byKey.get(key)).filter((policy): policy is BetSizingPolicyPreview => Boolean(policy)),
    frontierPolicies,
    byKey,
  };
}

export function selectDefaultPolicyKey(
  strategy: BetStrategy,
  strategyConfigs: Record<BetStrategy, ResolvedBetStrategyConfig>,
  frontierPolicies: BetSizingPolicyPreview[]
): string {
  const currentConfig = strategyConfigs[strategy];
  if (currentConfig?.config_signature) {
    return currentConfig.config_signature;
  }

  if (frontierPolicies.length > 0) {
    return frontierPolicies[0].configSignature;
  }

  return strategy;
}

export function selectBetSizingSlate(gamesData: GamesTodayResponse): BetSizingSlate {
  const allUpcomingRows = Array.isArray(gamesData.rows) ? gamesData.rows : [];
  const activeDateKey = gamesData.date_central || null;
  const upcomingRows =
    activeDateKey && allUpcomingRows.length > 0
      ? allUpcomingRows.filter((row) => dateKeyForScheduledGame(row) === activeDateKey)
      : allUpcomingRows;

  if (upcomingRows.length > 0) {
    const dateSummary = gamesData.date_central ? formatCentralDateSummary(gamesData.date_central) : "the current slate";
    return {
      label: `Using ${dateSummary} to explain today's sizing flow.`,
      rows: upcomingRows,
      source: "upcoming",
    };
  }

  const historicalRows = Array.isArray(gamesData.historical_rows) ? gamesData.historical_rows : [];
  if (historicalRows.length === 0) {
    return {
      label: "No current games or replay rows are available to illustrate the sizing flow.",
      rows: [],
      source: "historical",
    };
  }

  const latestDateKey =
    historicalRows
      .map((row) => dateKeyForScheduledGame(row))
      .filter((value): value is string => Boolean(value))
      .sort()
      .at(-1) || null;

  const rows = latestDateKey ? historicalRows.filter((row) => dateKeyForScheduledGame(row) === latestDateKey) : historicalRows;
  const label = latestDateKey
    ? `No upcoming slate is stored, so this view uses the latest replay slate from ${formatCentralDateLabel(latestDateKey)}.`
    : "No upcoming slate is stored, so this view uses the most recent replay rows.";

  return {
    label,
    rows,
    source: "historical",
  };
}

function compareGameRows(left: BetSizingGamePreview, right: BetSizingGamePreview): number {
  return (
    right.trace.finalStake - left.trace.finalStake ||
    (right.trace.candidateEdge ?? Number.NEGATIVE_INFINITY) - (left.trace.candidateEdge ?? Number.NEGATIVE_INFINITY) ||
    String(left.row.start_time_utc || "").localeCompare(String(right.row.start_time_utc || "")) ||
    left.row.game_id - right.row.game_id
  );
}

export function buildBetSizingGamePreviews(
  rows: GamesTodayRow[],
  strategy: BetStrategy,
  policy: BetSizingPolicyPreview
): BetSizingGamePreview[] {
  const traces = explainBetDecisionsForSlate(
    rows.map((row) => ({
      home_team: row.home_team,
      away_team: row.away_team,
      home_win_probability: row.home_win_probability,
      home_moneyline: row.home_moneyline,
      away_moneyline: row.away_moneyline,
      betting_model_name: row.betting_model_name,
      model_win_probabilities: row.model_win_probabilities,
    })),
    strategy,
    {
      allowUnderdogs: policy.allowUnderdogs,
      minEdge: policy.minEdge,
      minExpectedValue: policy.minExpectedValue,
      stakeScale: policy.stakeScale,
      maxBetBankrollPercent: policy.maxBetBankrollPercent,
      maxDailyBankrollPercent: policy.maxDailyBankrollPercent,
    },
    policy.label
  );

  return rows
    .map((row, index) => ({
      row,
      trace: traces[index],
    }))
    .sort(compareGameRows);
}

export function selectDefaultGameId(previews: BetSizingGamePreview[]): number | null {
  const firstBet = previews.find((preview) => preview.trace.finalStake > 0);
  return firstBet?.row.game_id ?? previews[0]?.row.game_id ?? null;
}
