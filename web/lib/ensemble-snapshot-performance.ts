import type { EnsembleSnapshotRow } from "@/lib/types";

const PERFORMANCE_TIMELINE_EXCLUDED_ACTIVATION_DATES = new Set(["2026-03-04"]);

export function filterPerformanceTimelineSnapshots(snapshots: EnsembleSnapshotRow[]): EnsembleSnapshotRow[] {
  return snapshots.filter((snapshot) => !PERFORMANCE_TIMELINE_EXCLUDED_ACTIVATION_DATES.has(snapshot.activation_date_central));
}
