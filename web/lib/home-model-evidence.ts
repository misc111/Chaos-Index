import type { CurrentBestTopModelSummary, EvidenceStatusSummary } from "@/lib/types";

export type HomeModelEvidenceTone = "research" | "blocked" | "review";

export type HomeModelEvidenceNote = {
  tone: HomeModelEvidenceTone;
  reason: string;
  rank?: number | null;
};

export type HomeEvidencePayload = {
  latest_artifact_role?: string | null;
  promotion_eligible?: boolean | null;
  production_ready?: boolean | null;
  evidence_status?: EvidenceStatusSummary | null;
  current_best_top_models?: CurrentBestTopModelSummary[];
};

function titleCaseToken(value?: string | null): string {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function normalizeReason(value?: string | null): string {
  return titleCaseToken(value)
    .replace(/\bMlb\b/g, "MLB")
    .replace(/\bCv\b/g, "CV")
    .replace(/\bGlm\b/g, "GLM");
}

function firstBlockedReason(evidence: HomeEvidencePayload): string {
  const readinessReasons = evidence.evidence_status?.readiness_blocked_reasons || [];
  const blockedReasons = evidence.evidence_status?.blocked_reasons || [];
  return normalizeReason(readinessReasons[0] || blockedReasons[0]) || "Evidence gate still open";
}

export function indexCurrentBestTopModels(rows?: CurrentBestTopModelSummary[]): Map<string, CurrentBestTopModelSummary> {
  return new Map((rows || []).filter((row) => row.model_name).map((row) => [String(row.model_name), row]));
}

export function buildHomeModelEvidenceBadge(
  modelKey: string,
  evidence: HomeEvidencePayload,
  topModels = indexCurrentBestTopModels(evidence.current_best_top_models)
): HomeModelEvidenceNote {
  const topModel = topModels.get(modelKey);
  const artifactRole = evidence.latest_artifact_role || evidence.evidence_status?.latest_artifact_role;
  const stage = evidence.evidence_status?.evidence_stage;

  if (topModel && (evidence.promotion_eligible || evidence.production_ready)) {
    return {
      tone: "review",
      reason: topModel.rank === 1 ? "Ready for a closer look." : "In the review mix.",
      rank: topModel.rank,
    };
  }

  if (topModel && (artifactRole === "latest_research_recommendation" || stage === "research_only")) {
    return {
      tone: "research",
      reason: topModel.rank === 1 ? "Showing promise, but not trusted for picks yet." : "Still earning trust in live testing.",
      rank: topModel.rank,
    };
  }

  if (topModel) {
    return {
      tone: "blocked",
      reason:
        firstBlockedReason(evidence) === "Not Full Immutable Pregame MLB Ledger"
          ? "Needs a clean pregame track record."
          : "Needs more proof before it gets trusted.",
      rank: topModel.rank,
    };
  }

  return {
    tone: "research",
    reason: "Waiting for enough fresh game proof.",
  };
}
