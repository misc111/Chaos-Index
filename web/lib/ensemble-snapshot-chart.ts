import { HISTORICAL_BANKROLL_START_DOLLARS } from "@/lib/betting";
import type { EnsembleSnapshotRow } from "@/lib/types";

export type SnapshotChartStrategyKey = "riskAdjusted" | "aggressive";

export type SnapshotBankrollPoint = {
  date_central: string;
  kind: "start" | "daily" | "pending";
  slate_games: number;
  suggested_bets: number;
  daily_profit: number;
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
  total_profit: number;
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

export function buildEnsembleSnapshotBankrollSeries(
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
        total_profit: strategySummary.total_profit,
        total_risked: strategySummary.total_risked,
        days_tracked: snapshot.days_tracked,
        points,
        final_point: finalPoint,
      };
    });
}

export function listEnsembleSnapshotChartDates(series: SnapshotBankrollSeries[]): string[] {
  return Array.from(new Set(series.flatMap((snapshot) => snapshot.points.map((point) => point.date_central)))).sort();
}
