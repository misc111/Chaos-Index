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

  return NextResponse.json({
    league,
    current_best: currentBest || null,
    summary: summary || null,
    target_coverage: Array.isArray(summary?.targets) ? summary?.targets : [],
    family_champions: Array.isArray(summary?.family_champions) ? summary?.family_champions : [],
    inter_family_leaderboard: Array.isArray(summary?.inter_family_leaderboard) ? summary?.inter_family_leaderboard : [],
    artifacts: summary?.artifacts || {},
  });
}
