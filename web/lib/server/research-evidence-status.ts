import fs from "node:fs";
import path from "node:path";

import type { LeagueCode } from "@/lib/league";
import type { CurrentBestTopModelSummary, ResearchDeskResponse, TableRow } from "@/lib/types";

function parseJsonRecord(value?: string | null): TableRow | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function readJsonRecord(filePath: string): TableRow | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function readRecordArray(value: unknown): TableRow[] {
  return Array.isArray(value)
    ? value.filter((row): row is TableRow => Boolean(row) && typeof row === "object" && !Array.isArray(row))
    : [];
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function summarizeTopModel(row: TableRow): CurrentBestTopModelSummary {
  return {
    model_name: typeof row.model_name === "string" ? row.model_name : null,
    display_name: typeof row.display_name === "string" ? row.display_name : null,
    rank: readNumber(row.rank),
    promotion_role: typeof row.promotion_role === "string" ? row.promotion_role : null,
    model_lane: typeof row.model_lane === "string" ? row.model_lane : null,
    evidence_lane: typeof row.evidence_lane === "string" ? row.evidence_lane : null,
    theory_classification: typeof row.theory_classification === "string" ? row.theory_classification : null,
    theory_governance: typeof row.theory_governance === "string" ? row.theory_governance : null,
    governance_note: typeof row.governance_note === "string" ? row.governance_note : null,
    mean_auc: readNumber(row.mean_auc),
    mean_log_loss: readNumber(row.mean_log_loss),
    mean_brier: readNumber(row.mean_brier),
    mean_ece: readNumber(row.mean_ece),
    mean_roi: readNumber(row.mean_roi),
    final_holdout_log_loss: readNumber(row.final_holdout_log_loss),
    final_holdout_brier: readNumber(row.final_holdout_brier),
    final_holdout_auc: readNumber(row.final_holdout_auc),
    final_holdout_ece: readNumber(row.final_holdout_ece),
    validation_log_loss: readNumber(row.validation_log_loss),
    validation_brier: readNumber(row.validation_brier),
    validation_auc: readNumber(row.validation_auc),
    bet_count: readNumber(row.bet_count),
    margin_diagnostics:
      row.margin_diagnostics && typeof row.margin_diagnostics === "object" && !Array.isArray(row.margin_diagnostics)
        ? (row.margin_diagnostics as TableRow)
        : null,
  };
}

export function loadEvidenceStatus(league: LeagueCode): Pick<
  ResearchDeskResponse,
  | "source_kind"
  | "source_status"
  | "evidence_stage"
  | "latest_artifact_role"
  | "promotion_eligible"
  | "production_ready"
  | "evidence_status"
  | "current_best_top_models"
> {
  if (league !== "MLB") {
    return {
      source_kind: null,
      source_status: null,
      evidence_stage: null,
      latest_artifact_role: null,
      promotion_eligible: false,
      production_ready: false,
      evidence_status: null,
      current_best_top_models: [],
    };
  }
  const repoRoot = path.resolve(process.cwd(), "..");
  const currentBest = readJsonRecord(path.resolve(repoRoot, "artifacts", "reports", league.toLowerCase(), "current_best_models.json"));
  const evidenceStatus = parseJsonRecord(JSON.stringify(currentBest?.evidence_status ?? null));
  const latestArtifactRole =
    typeof currentBest?.latest_artifact_role === "string" ? String(currentBest.latest_artifact_role) : null;
  const evidenceStage =
    typeof evidenceStatus?.evidence_stage === "string" ? String(evidenceStatus.evidence_stage) : null;
  return {
    source_kind: typeof currentBest?.source_kind === "string" ? String(currentBest.source_kind) : null,
    source_status: typeof currentBest?.source_status === "string" ? String(currentBest.source_status) : null,
    evidence_stage: evidenceStage,
    latest_artifact_role: latestArtifactRole,
    promotion_eligible: Boolean(currentBest?.promotion_eligible === true),
    production_ready: Boolean(currentBest?.production_ready === true || evidenceStatus?.production_ready === true),
    evidence_status: evidenceStatus,
    current_best_top_models: readRecordArray(currentBest?.top_models).map(summarizeTopModel),
  };
}
