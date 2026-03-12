import type { ModelReplayDecisionDetail, EnsembleSnapshotDailyRow, EnsembleSnapshotRow, PerformanceReplayExperimentSummary } from "@/lib/types";

type DailyCumulativeState = {
  total_profit: number;
  total_risked: number;
};

type MutableStrategySummary = {
  total_games: number;
  suggested_bets: number;
  wins: number;
  losses: number;
  total_risked: number;
  total_profit: number;
  first_bet_date_central: string | null;
  last_bet_date_central: string | null;
  edge_sum: number;
  edge_count: number;
  expected_value_sum: number;
  expected_value_count: number;
};

type PerformanceReplayExperimentDefinition = PerformanceReplayExperimentSummary & {
  max_age_days: number;
  max_underdog_odds: number;
};

const PERFORMANCE_REPLAY_EXPERIMENTS: Record<string, PerformanceReplayExperimentDefinition> = {
  "fresh-1d-no-dogs-over-300": {
    id: "fresh-1d-no-dogs-over-300",
    label: "Fresh <= 1 day, no dogs > +300",
    description:
      "Keeps the same strategy thresholds and sizing, but zeroes any replay bet more than 1 day after the snapshot as-of date and any underdog priced above +300.",
    scope: "ensemble_snapshots",
    max_age_days: 1,
    max_underdog_odds: 300,
  },
};

export type PerformanceReplayExperimentId = keyof typeof PERFORMANCE_REPLAY_EXPERIMENTS;

export function normalizePerformanceReplayExperiment(value?: string | null): PerformanceReplayExperimentId | null {
  const token = String(value || "").trim().toLowerCase();
  if (token in PERFORMANCE_REPLAY_EXPERIMENTS) {
    return token as PerformanceReplayExperimentId;
  }
  return null;
}

export function performanceReplayExperimentFromRequest(request: Request): PerformanceReplayExperimentId | null {
  const url = new URL(request.url);
  return normalizePerformanceReplayExperiment(url.searchParams.get("experiment"));
}

export function getPerformanceReplayExperimentSummary(
  value?: string | null
): PerformanceReplayExperimentSummary | null {
  const normalized = normalizePerformanceReplayExperiment(value);
  if (!normalized) return null;
  const experiment = PERFORMANCE_REPLAY_EXPERIMENTS[normalized];
  return {
    id: experiment.id,
    label: experiment.label,
    description: experiment.description,
    scope: experiment.scope,
  };
}

export function listPerformanceReplayExperiments(): PerformanceReplayExperimentSummary[] {
  return Object.values(PERFORMANCE_REPLAY_EXPERIMENTS).map((experiment) => ({
    id: experiment.id,
    label: experiment.label,
    description: experiment.description,
    scope: experiment.scope,
  }));
}

export function buildPerformanceExperimentStagingFileName(value?: string | null): string {
  const normalized = normalizePerformanceReplayExperiment(value);
  return normalized ? `performance.${normalized}.json` : "performance.json";
}

function parseDateKey(value?: string | null): Date | null {
  const raw = String(value || "").trim();
  if (!raw) return null;
  const parsed = new Date(raw.includes("T") ? raw : `${raw}T12:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function differenceInDays(left?: string | null, right?: string | null): number | null {
  const leftDate = parseDateKey(left);
  const rightDate = parseDateKey(right);
  if (!leftDate || !rightDate) return null;
  const millisecondsPerDay = 24 * 60 * 60 * 1000;
  return Math.round((rightDate.getTime() - leftDate.getTime()) / millisecondsPerDay);
}

function zeroOutDecision(detail: ModelReplayDecisionDetail, reason: string): ModelReplayDecisionDetail {
  return {
    ...detail,
    bet_label: "$0",
    reason,
    stake: 0,
    outcome: "no_bet",
    profit: 0,
    payout: 0,
  };
}

function applyExperimentToDecision(
  detail: ModelReplayDecisionDetail,
  betDateCentral: string,
  anchorDateCentral: string | null,
  experiment: PerformanceReplayExperimentDefinition
): ModelReplayDecisionDetail {
  if (detail.stake <= 0 || detail.outcome === "no_bet") {
    return detail;
  }

  const reasons: string[] = [];
  const ageInDays = differenceInDays(anchorDateCentral, betDateCentral);
  if (ageInDays !== null && ageInDays > experiment.max_age_days) {
    reasons.push(`Experiment skips bets more than ${experiment.max_age_days} day${experiment.max_age_days === 1 ? "" : "s"} after the snapshot date`);
  }
  if (typeof detail.odds === "number" && detail.odds > experiment.max_underdog_odds) {
    reasons.push(`Experiment skips underdogs above +${experiment.max_underdog_odds}`);
  }

  return reasons.length > 0 ? zeroOutDecision(detail, reasons.join("; ")) : detail;
}

function emptyStrategySummary(totalGames: number): MutableStrategySummary {
  return {
    total_games: totalGames,
    suggested_bets: 0,
    wins: 0,
    losses: 0,
    total_risked: 0,
    total_profit: 0,
    first_bet_date_central: null,
    last_bet_date_central: null,
    edge_sum: 0,
    edge_count: 0,
    expected_value_sum: 0,
    expected_value_count: 0,
  };
}

function trackStrategyOutcome(
  summary: MutableStrategySummary,
  detail: ModelReplayDecisionDetail,
  dateCentral: string
): void {
  if (detail.stake <= 0 || detail.outcome === "no_bet") {
    return;
  }

  summary.suggested_bets += 1;
  summary.total_risked += detail.stake;
  summary.total_profit += detail.profit;
  if (detail.outcome === "win") summary.wins += 1;
  if (detail.outcome === "loss") summary.losses += 1;
  if (typeof detail.edge === "number" && Number.isFinite(detail.edge)) {
    summary.edge_sum += detail.edge;
    summary.edge_count += 1;
  }
  if (typeof detail.expected_value === "number" && Number.isFinite(detail.expected_value)) {
    summary.expected_value_sum += detail.expected_value;
    summary.expected_value_count += 1;
  }
  if (!summary.first_bet_date_central || dateCentral < summary.first_bet_date_central) {
    summary.first_bet_date_central = dateCentral;
  }
  if (!summary.last_bet_date_central || dateCentral > summary.last_bet_date_central) {
    summary.last_bet_date_central = dateCentral;
  }
}

function finalizeStrategySummary(summary: MutableStrategySummary) {
  return {
    total_games: summary.total_games,
    suggested_bets: summary.suggested_bets,
    wins: summary.wins,
    losses: summary.losses,
    total_risked: summary.total_risked,
    total_profit: summary.total_profit,
    roi: summary.total_risked > 0 ? summary.total_profit / summary.total_risked : 0,
    avg_edge: summary.edge_count > 0 ? summary.edge_sum / summary.edge_count : null,
    avg_expected_value: summary.expected_value_count > 0 ? summary.expected_value_sum / summary.expected_value_count : null,
    first_bet_date_central: summary.first_bet_date_central,
    last_bet_date_central: summary.last_bet_date_central,
  };
}

function summarizeDetailsForDate(
  details: ModelReplayDecisionDetail[],
  dateCentral: string,
  totalGames: number
): MutableStrategySummary {
  const summary = emptyStrategySummary(totalGames);
  for (const detail of details) {
    trackStrategyOutcome(summary, detail, dateCentral);
  }
  return summary;
}

function buildDailyStrategyRow(
  details: ModelReplayDecisionDetail[],
  dateCentral: string,
  slateGames: number,
  cumulativeState: DailyCumulativeState
): EnsembleSnapshotDailyRow["strategies"]["riskAdjusted"] {
  const daySummary = summarizeDetailsForDate(details, dateCentral, slateGames);
  cumulativeState.total_profit += daySummary.total_profit;
  cumulativeState.total_risked += daySummary.total_risked;

  return {
    slate_games: slateGames,
    suggested_bets: daySummary.suggested_bets,
    wins: daySummary.wins,
    losses: daySummary.losses,
    total_risked: daySummary.total_risked,
    total_profit: daySummary.total_profit,
    cumulative_risked: cumulativeState.total_risked,
    cumulative_profit: cumulativeState.total_profit,
    roi: daySummary.total_risked > 0 ? daySummary.total_profit / daySummary.total_risked : 0,
    cumulative_roi:
      cumulativeState.total_risked > 0 ? cumulativeState.total_profit / cumulativeState.total_risked : 0,
  };
}

export function applyPerformanceReplayExperimentToEnsembleSnapshots(
  snapshots: EnsembleSnapshotRow[],
  value?: string | null
): EnsembleSnapshotRow[] {
  const normalized = normalizePerformanceReplayExperiment(value);
  if (!normalized) {
    return snapshots;
  }

  const experiment = PERFORMANCE_REPLAY_EXPERIMENTS[normalized];

  return snapshots.map((snapshot) => {
    const anchorDateCentral =
      snapshot.finalized_date_central || snapshot.activation_date_central || snapshot.daily[0]?.date_central || null;
    const bets = snapshot.bets.map((bet) => ({
      ...bet,
      strategies: {
        riskAdjusted: applyExperimentToDecision(bet.strategies.riskAdjusted, bet.date_central, anchorDateCentral, experiment),
        aggressive: applyExperimentToDecision(bet.strategies.aggressive, bet.date_central, anchorDateCentral, experiment),
        capitalPreservation: applyExperimentToDecision(
          bet.strategies.capitalPreservation,
          bet.date_central,
          anchorDateCentral,
          experiment
        ),
      },
    }));

    const totalGames = snapshot.replayable_games || bets.length;
    const riskAdjustedSummary = emptyStrategySummary(totalGames);
    const aggressiveSummary = emptyStrategySummary(totalGames);
    const capitalPreservationSummary = emptyStrategySummary(totalGames);

    for (const bet of bets) {
      trackStrategyOutcome(riskAdjustedSummary, bet.strategies.riskAdjusted, bet.date_central);
      trackStrategyOutcome(aggressiveSummary, bet.strategies.aggressive, bet.date_central);
      trackStrategyOutcome(capitalPreservationSummary, bet.strategies.capitalPreservation, bet.date_central);
    }

    const cumulativeState = {
      riskAdjusted: { total_profit: 0, total_risked: 0 },
      aggressive: { total_profit: 0, total_risked: 0 },
      capitalPreservation: { total_profit: 0, total_risked: 0 },
    };

    const daily = snapshot.daily.map((day) => {
      const dayBets = bets.filter((bet) => bet.date_central === day.date_central);
      return {
        ...day,
        strategies: {
          riskAdjusted: buildDailyStrategyRow(
            dayBets.map((bet) => bet.strategies.riskAdjusted),
            day.date_central,
            day.slate_games,
            cumulativeState.riskAdjusted
          ),
          aggressive: buildDailyStrategyRow(
            dayBets.map((bet) => bet.strategies.aggressive),
            day.date_central,
            day.slate_games,
            cumulativeState.aggressive
          ),
          capitalPreservation: buildDailyStrategyRow(
            dayBets.map((bet) => bet.strategies.capitalPreservation),
            day.date_central,
            day.slate_games,
            cumulativeState.capitalPreservation
          ),
        },
      };
    });

    return {
      ...snapshot,
      strategies: {
        riskAdjusted: finalizeStrategySummary(riskAdjustedSummary),
        aggressive: finalizeStrategySummary(aggressiveSummary),
        capitalPreservation: finalizeStrategySummary(capitalPreservationSummary),
      },
      daily,
      bets,
    };
  });
}
