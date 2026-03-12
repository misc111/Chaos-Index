import { HISTORICAL_BANKROLL_START_DOLLARS } from "@/lib/betting";
import type { BetStrategy } from "@/lib/betting-strategy";
import type { EnsembleSnapshotRow } from "@/lib/types";

export type SnapshotChartStrategyKey = BetStrategy;
export type SnapshotBankrollMode = "independent" | "continuity";

export type SnapshotBankrollPoint = {
  date_central: string;
  kind: "start" | "daily" | "pending";
  slate_games: number;
  suggested_bets: number;
  daily_profit: number;
  snapshot_cumulative_profit: number;
  cumulative_profit: number;
  cumulative_bankroll: number;
  total_risked: number;
};

export type SnapshotBankrollSeries = {
  snapshot_key: string;
  activation_date_central: string;
  compared_through_date_central: string | null;
  feature_set_version?: string | null;
  replayable_games: number;
  starting_bankroll: number;
  isolated_total_profit: number;
  display_total_profit: number;
  total_risked: number;
  days_tracked: number;
  points: SnapshotBankrollPoint[];
  final_point: SnapshotBankrollPoint;
};

export function previousDateKey(dateKey: string): string {
  const parsed = new Date(`${dateKey}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return dateKey;
  parsed.setUTCDate(parsed.getUTCDate() - 1);
  return parsed.toISOString().slice(0, 10);
}

function resolveStartingAnchorDate(snapshot: EnsembleSnapshotRow): string {
  const firstTrackedDate = snapshot.daily[0]?.date_central;
  if (!firstTrackedDate) {
    return snapshot.activation_date_central;
  }

  // When the snapshot immediately goes live on the same date as the first
  // settled replay row, anchor the bankroll path to the previous day so the
  // chart can show the opening bankroll before any bets settle.
  if (firstTrackedDate <= snapshot.activation_date_central) {
    return previousDateKey(firstTrackedDate);
  }

  // If there was a gap between activation and the first settled slate, anchor
  // on the activation date itself so the line honestly shows when that frozen
  // model started "living forever" in the counterfactual replay.
  return snapshot.activation_date_central;
}

function buildIndependentEnsembleSnapshotBankrollSeries(
  snapshots: EnsembleSnapshotRow[],
  strategy: SnapshotChartStrategyKey
): SnapshotBankrollSeries[] {
  return snapshots
    .slice()
    .sort((left, right) => left.activation_date_central.localeCompare(right.activation_date_central))
    .map((snapshot) => {
      const strategySummary = snapshot.strategies[strategy];
      const startingPoint: SnapshotBankrollPoint = {
        date_central: resolveStartingAnchorDate(snapshot),
        kind: snapshot.daily.length ? "start" : "pending",
        slate_games: 0,
        suggested_bets: 0,
        daily_profit: 0,
        snapshot_cumulative_profit: 0,
        cumulative_profit: 0,
        cumulative_bankroll: HISTORICAL_BANKROLL_START_DOLLARS,
        total_risked: 0,
      };

      const dailyPoints: SnapshotBankrollPoint[] = snapshot.daily.map((day) => ({
        date_central: day.date_central,
        kind: "daily",
        slate_games: day.slate_games,
        suggested_bets: day.strategies[strategy].suggested_bets,
        daily_profit: day.strategies[strategy].total_profit,
        snapshot_cumulative_profit: day.strategies[strategy].cumulative_profit,
        cumulative_profit: day.strategies[strategy].cumulative_profit,
        cumulative_bankroll: HISTORICAL_BANKROLL_START_DOLLARS + day.strategies[strategy].cumulative_profit,
        total_risked: day.strategies[strategy].total_risked,
      }));

      const points = [startingPoint, ...dailyPoints];
      const finalPoint = points[points.length - 1] || startingPoint;

      return {
        snapshot_key: snapshot.snapshot_key,
        activation_date_central: snapshot.activation_date_central,
        compared_through_date_central: snapshot.compared_through_date_central,
        feature_set_version: snapshot.feature_set_version,
        replayable_games: snapshot.replayable_games,
        starting_bankroll: HISTORICAL_BANKROLL_START_DOLLARS,
        isolated_total_profit: strategySummary.total_profit,
        display_total_profit: strategySummary.total_profit,
        total_risked: strategySummary.total_risked,
        days_tracked: snapshot.days_tracked,
        points,
        final_point: finalPoint,
      };
    });
}

function bankrollThroughDate(series: SnapshotBankrollSeries, dateCentral: string): number | null {
  let bankroll: number | null = null;
  for (const point of series.points) {
    if (point.date_central > dateCentral) break;
    bankroll = point.cumulative_bankroll;
  }
  return bankroll;
}

function applyContinuityToSeries(series: SnapshotBankrollSeries[]): SnapshotBankrollSeries[] {
  const shiftedSeries: SnapshotBankrollSeries[] = [];

  for (const snapshot of series) {
    const previousSnapshot = shiftedSeries[shiftedSeries.length - 1];
    const handoffDate = previousDateKey(snapshot.activation_date_central);
    const startingBankroll =
      previousSnapshot && shiftedSeries.length
        ? bankrollThroughDate(previousSnapshot, handoffDate) ?? previousSnapshot.final_point.cumulative_bankroll
        : HISTORICAL_BANKROLL_START_DOLLARS;
    const bankrollOffset = startingBankroll - HISTORICAL_BANKROLL_START_DOLLARS;

    // Continuity mode does not change the replayed bets themselves. It only
    // changes the bankroll basis so each newly activated snapshot inherits the
    // bankroll level that the prior deployed snapshot had reached by D-1.
    const shiftedPoints = snapshot.points.map((point) => ({
      ...point,
      cumulative_profit: point.cumulative_profit + bankrollOffset,
      cumulative_bankroll: point.cumulative_bankroll + bankrollOffset,
    }));
    const finalPoint = shiftedPoints[shiftedPoints.length - 1] || shiftedPoints[0];

    shiftedSeries.push({
      ...snapshot,
      starting_bankroll: startingBankroll,
      display_total_profit: (finalPoint?.cumulative_bankroll ?? startingBankroll) - HISTORICAL_BANKROLL_START_DOLLARS,
      points: shiftedPoints,
      final_point: finalPoint,
    });
  }

  return shiftedSeries;
}

export function resolveSnapshotAccountBankrollOnDate(
  series: SnapshotBankrollSeries[],
  dateCentral: string
): number | null {
  let bankroll: number | null = null;

  for (const snapshot of series) {
    const firstPointDate = snapshot.points[0]?.date_central;
    if (firstPointDate && firstPointDate > dateCentral) {
      break;
    }

    const candidate = bankrollThroughDate(snapshot, dateCentral);
    if (candidate !== null) {
      bankroll = candidate;
    }
  }

  return bankroll;
}

export function buildEnsembleSnapshotBankrollSeries(
  snapshots: EnsembleSnapshotRow[],
  strategy: SnapshotChartStrategyKey,
  mode: SnapshotBankrollMode = "independent"
): SnapshotBankrollSeries[] {
  const independentSeries = buildIndependentEnsembleSnapshotBankrollSeries(snapshots, strategy);
  return mode === "continuity" ? applyContinuityToSeries(independentSeries) : independentSeries;
}

export function listEnsembleSnapshotChartDates(series: SnapshotBankrollSeries[]): string[] {
  return Array.from(new Set(series.flatMap((snapshot) => snapshot.points.map((point) => point.date_central)))).sort();
}
