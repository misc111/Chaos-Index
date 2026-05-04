import { NextResponse } from "next/server.js";
import fs from "node:fs";
import path from "node:path";
import { leagueFromRequest } from "@/lib/league";

export const dynamic = "force-dynamic";

function readJson(filePath: string): Record<string, unknown> | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function latestNestedSummary(root: string): Record<string, unknown> | null {
  const nestedRoot = path.join(root, "tournament", "nested");
  if (!fs.existsSync(nestedRoot)) return null;
  const runDirs = fs
    .readdirSync(nestedRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(nestedRoot, entry.name))
    .sort((left, right) => right.localeCompare(left));
  for (const runDir of runDirs) {
    const summary = readJson(path.join(runDir, "nested_tournament_summary.json"));
    if (summary) return summary;
  }
  return null;
}

function isNestedTournamentSourceKind(value: unknown): boolean {
  return value === "nested_mlb_model_tournament" || value === "parallel_nested_mlb_model_tournament";
}

function resolveRepoArtifactPath(repoRoot: string, artifactPath: string): string {
  return path.isAbsolute(artifactPath) ? artifactPath : path.resolve(repoRoot, artifactPath);
}

function nestedArtifact(summary: Record<string, unknown> | null, key: string): string {
  const artifacts = summary?.artifacts;
  if (!artifacts || typeof artifacts !== "object" || Array.isArray(artifacts)) return "";
  return String((artifacts as Record<string, unknown>)[key] || "");
}

function nestedResearchEvidenceStatus(summary: Record<string, unknown> | null): Record<string, unknown> {
  const summaryEvidence =
    summary?.evidence_status && typeof summary.evidence_status === "object" && !Array.isArray(summary.evidence_status)
      ? (summary.evidence_status as Record<string, unknown>)
      : null;
  return {
    source_kind: summary?.evidence_scope || "nested_mlb_model_tournament",
    source_status: "research_only",
    evidence_stage: String(summaryEvidence?.evidence_stage || "").trim() || "research_only",
    latest_artifact_role: "latest_research_recommendation",
    promotion_eligible: false,
    production_ready: false,
    evidence_status:
      summaryEvidence || {
        evidence_stage: String(summary?.run_id || "").toLowerCase().includes("smoke") ? "fixture_demo_smoke" : "research_only",
        label: "nested tournament research evidence",
        latest_artifact_role: "latest_research_recommendation",
        pointer_semantics:
          "This pointer identifies the latest nested tournament research evidence. It is not full-ledger promotion-eligible evidence.",
        promotion_eligible: false,
        production_grade: false,
        production_ready: false,
        fixture_demo_smoke: String(summary?.run_id || "").toLowerCase().includes("smoke"),
        full_immutable_pregame_ledger: false,
        blocked_reasons: ["nested_tournament_research_only", "not_full_immutable_pregame_mlb_ledger"],
      },
    latest_research_recommendation: {
      run_id: summary?.run_id || null,
      source_kind: summary?.evidence_scope || "nested_mlb_model_tournament",
      pointer_semantics: "Latest nested tournament research evidence only; not promotion-eligible evidence.",
      artifact_path: nestedArtifact(summary, "summary_path") || null,
    },
    latest_promotion_eligible_evidence: null,
  };
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const repoRoot = path.resolve(process.cwd(), "..");
  const reportRoot = path.resolve(repoRoot, "artifacts", "reports", league.toLowerCase());
  const currentBest = readJson(path.join(reportRoot, "current_best_models.json"));
  const summaryPath =
    currentBest &&
    typeof currentBest === "object" &&
    isNestedTournamentSourceKind(currentBest.source_kind) &&
    currentBest.artifact_paths &&
    typeof currentBest.artifact_paths === "object"
      ? String((currentBest.artifact_paths as Record<string, unknown>).summary_path || "")
      : "";
  const summary = summaryPath ? readJson(resolveRepoArtifactPath(repoRoot, summaryPath)) : latestNestedSummary(reportRoot);
  const featureCoverageJson = nestedArtifact(summary, "feature_coverage_by_split_json");
  const featureCoverage = featureCoverageJson ? readJson(resolveRepoArtifactPath(repoRoot, featureCoverageJson)) : null;
  const nestedCurrentBest =
    currentBest && isNestedTournamentSourceKind(currentBest.source_kind) ? currentBest : nestedResearchEvidenceStatus(summary);

  return NextResponse.json({
    league,
    current_best: summary ? nestedCurrentBest : null,
    summary: summary || null,
    target_coverage: Array.isArray(summary?.targets) ? summary?.targets : [],
    family_champions: Array.isArray(summary?.family_champions) ? summary?.family_champions : [],
    inter_family_leaderboard: Array.isArray(summary?.inter_family_leaderboard) ? summary?.inter_family_leaderboard : [],
    feature_coverage_summary: Array.isArray(featureCoverage?.summary) ? featureCoverage?.summary : [],
    artifacts: summary?.artifacts || {},
  });
}
