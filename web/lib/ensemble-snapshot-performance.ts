import type { EnsembleSnapshotRow } from "@/lib/types";

// Keep the legacy March 4 snapshot off the Performance tab. We still use
// March 4 as the bankroll reference date, but the snapshot card/timeline entry
// itself should stay hidden from this surface.
const PERFORMANCE_TIMELINE_EXCLUDED_ACTIVATION_DATES = new Set(["2026-03-04"]);

export function filterPerformanceTimelineSnapshots(snapshots: EnsembleSnapshotRow[]): EnsembleSnapshotRow[] {
  return snapshots.filter((snapshot) => !PERFORMANCE_TIMELINE_EXCLUDED_ACTIVATION_DATES.has(snapshot.activation_date_central));
}
