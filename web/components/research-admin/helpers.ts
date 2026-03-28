import type {
  ResearchAdminResponse,
  ResearchChampionSummary,
  ResearchPromotionSummary,
  ResearchRunRow,
  TableRow,
} from "@/lib/types";

export type ChampionTimelineEvent = {
  key: string;
  model_name: string;
  occurred_at_utc: string;
  label: string;
  source_run_id: string | null;
  source_brief_id: string | null;
  reason_summary: string | null;
  is_active: boolean;
};

function asObject(value: unknown): TableRow | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as TableRow) : null;
}

function firstScalar(record: TableRow | null, keys: string[]): string | null {
  if (!record) return null;
  for (const key of keys) {
    const value = record[key];
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
  }
  return null;
}

function compactNumber(value: unknown): string | null {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  if (Number.isInteger(numeric)) {
    return String(numeric);
  }
  return numeric.toFixed(Math.abs(numeric) >= 10 ? 0 : 2);
}

export function formatAdminTimestamp(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "America/Chicago",
  });
}

export function buildChampionTimeline(
  champion: ResearchChampionSummary | null,
  decisions: ResearchPromotionSummary[]
): ChampionTimelineEvent[] {
  const byKey = new Map<string, ChampionTimelineEvent>();

  if (champion?.model_name && champion.promoted_at_utc) {
    const key = `${champion.model_name}::${champion.promoted_at_utc}`;
    byKey.set(key, {
      key,
      model_name: champion.model_name,
      occurred_at_utc: champion.promoted_at_utc,
      label: "Active champion",
      source_run_id: champion.source_run_id ?? null,
      source_brief_id: champion.source_brief_id ?? null,
      reason_summary: null,
      is_active: true,
    });
  }

  for (const decision of decisions) {
    if (!decision.promoted || !decision.candidate_model_name || !decision.created_at_utc) {
      continue;
    }
    const key = `${decision.candidate_model_name}::${decision.created_at_utc}`;
    const existing = byKey.get(key);
    byKey.set(key, {
      key,
      model_name: decision.candidate_model_name,
      occurred_at_utc: decision.created_at_utc,
      label: existing?.is_active ? existing.label : "Promotion approved",
      source_run_id: existing?.source_run_id ?? null,
      source_brief_id: existing?.source_brief_id ?? null,
      reason_summary: decision.reason_summary ?? existing?.reason_summary ?? null,
      is_active: existing?.is_active ?? false,
    });
  }

  return Array.from(byKey.values()).sort((left, right) => right.occurred_at_utc.localeCompare(left.occurred_at_utc));
}

export function summarizeRunSummary(summary?: TableRow | null): string {
  const record = asObject(summary);
  if (!record) return "No stored summary.";

  const headline =
    firstScalar(record, ["headline", "summary", "status_message", "reason_summary", "best_candidate_model"]) ?? null;

  const metrics = [
    ["bankroll", compactNumber(record.bankroll)],
    ["drawdown", compactNumber(record.max_drawdown)],
    ["profitable folds", compactNumber(record.profitable_folds)],
    ["bet count", compactNumber(record.bet_count)],
  ]
    .filter(([, value]) => value)
    .map(([label, value]) => `${label} ${value}`);

  return [headline, ...metrics].filter(Boolean).join(" · ") || "Stored summary available.";
}

export function describeChampion(champion: ResearchChampionSummary | null): string {
  if (!champion) {
    return "No champion promoted yet.";
  }
  const descriptor = asObject(champion.descriptor);
  const candidateCount = compactNumber(descriptor?.candidate_count);
  const source = champion.source_run_id ? `Run ${champion.source_run_id}` : "Manual fallback state";
  return candidateCount ? `${source} · ${candidateCount} candidate models reviewed` : source;
}

export function buildAdminCounts(payload: ResearchAdminResponse) {
  return {
    briefs: payload.briefs.length,
    runs: payload.runs.length,
    decisions: payload.decisions.length,
    promotions: payload.decisions.filter((decision) => decision.promoted).length,
  };
}

export function latestRunForChampion(payload: ResearchAdminResponse): ResearchRunRow | null {
  if (!payload.champion?.source_run_id) return null;
  return payload.runs.find((run) => run.run_id === payload.champion?.source_run_id) ?? null;
}
