import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DASHBOARD_STAGING_ROUTES } from "../lib/generated/dashboard-routes";
import { type LeagueCode } from "../lib/generated/league-registry";
import {
  buildPerformanceExperimentStagingFileName,
  listPerformanceReplayExperiments,
} from "../lib/performance-replay-experiments";
import { STAGING_ROUTE_LOADERS, type JsonRouteHandler } from "./staging-route-loaders";
import {
  buildStagingManifestPayload,
  PRIMARY_STAGING_LEAGUE,
  SHIPPED_STAGING_LEAGUES,
} from "./staging-contract";

type JsonRecord = Record<string, unknown>;

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(scriptDir, "..");
// Maintainer note: this script is the bridge from the live local dashboard
// data model to the committed GitHub Pages staging snapshot. If a local
// dashboard/API change should be visible on staging, rerun this script and
// commit the resulting files under web/public/staging-data/.
const outputRoot = path.join(appRoot, "public", "staging-data");
const generatedAtUtc = new Date().toISOString();

const PUBLIC_VALIDATION_SECTIONS = [
  "split_summary",
  "glm_residual_summary",
  "significance",
  "information_criteria_summary",
  "cv_summary",
  "collinearity_summary",
  "nonlinearity_summary",
  "calibration_robustness",
  "market_truth_summary",
  "logit_quantile_summary",
  "logit_lift_summary",
  "logit_roc_summary",
] as const;

function requestForLeague(routePath: string, league: LeagueCode, params: Record<string, string> = {}): Request {
  const url = new URL(`http://staging.local${routePath}`);
  url.searchParams.set("league", league);
  for (const [key, value] of Object.entries(params)) {
    if (value) {
      url.searchParams.set(key, value);
    }
  }
  return new Request(url);
}

function asRecordArray(value: unknown): JsonRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((row): row is JsonRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row));
}

function sanitizePublicPayload(fileName: string, payload: unknown, league: LeagueCode): unknown {
  if (fileName === "nested-tournament.json") {
    const raw = (payload || {}) as JsonRecord;
    const championRows = asRecordArray(raw.inter_family_leaderboard).filter((row) => Number(row.target_rank) <= 3);
    const familyRows = asRecordArray(raw.family_champions).filter((row) => row.family_champion === true || row.family_champion === 1);
    const rawArtifacts = ((raw.artifacts as JsonRecord | undefined) || {}) as JsonRecord;
    return {
      league,
      summary: raw.summary ? { run_id: (raw.summary as JsonRecord).run_id, generated_at_utc: (raw.summary as JsonRecord).generated_at_utc } : null,
      target_coverage: asRecordArray(raw.target_coverage),
      family_champions: familyRows,
      inter_family_leaderboard: championRows,
      feature_coverage_summary: asRecordArray(raw.feature_coverage_summary),
      artifacts: {
        feature_coverage_by_split_path: rawArtifacts.feature_coverage_by_split_path ?? null,
        feature_coverage_by_split_json: rawArtifacts.feature_coverage_by_split_json ?? null,
        validation_autopsy_report: rawArtifacts.validation_autopsy_report ?? null,
        gate_failure_counts_path: rawArtifacts.gate_failure_counts_path ?? null,
      },
      public_note:
        "Compact nested tournament digest. High-volume per-variant diagnostics stay local; champion/top-challenger diagnostic artifact references are retained.",
    };
  }

  if (fileName !== "validation.json") {
    return payload;
  }

  const raw = (payload || {}) as JsonRecord;
  const rawSections = ((raw.sections as JsonRecord | undefined) || {}) as JsonRecord;
  const sections = Object.fromEntries(
    PUBLIC_VALIDATION_SECTIONS.map((section) => [section, asRecordArray(rawSections[section])]).filter(([, rows]) => rows.length > 0)
  ) as Record<string, JsonRecord[]>;
  sections.public_note = [
    {
      message:
        "Compact public validation digest published from the local artifact set. Detailed coefficient paths, residual bins, and other high-volume tables stay private in GitHub Pages staging.",
    },
  ];

  return {
    league,
    significance: asRecordArray(raw.significance).length ? asRecordArray(raw.significance) : asRecordArray(rawSections.significance),
    sections,
    as_of_utc: raw.as_of_utc ?? null,
  };
}

async function writeJson(filePath: string, payload: unknown): Promise<void> {
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

async function loadRouteHandler(modulePath: string): Promise<JsonRouteHandler> {
  const routeLoader = STAGING_ROUTE_LOADERS[modulePath];
  if (typeof routeLoader !== "function") {
    throw new Error(`No route loader configured for ${modulePath}`);
  }
  return await routeLoader();
}

async function generateLeagueSnapshot(league: LeagueCode): Promise<void> {
  const leagueDir = path.join(outputRoot, league.toLowerCase());
  await fs.mkdir(leagueDir, { recursive: true });
  const generatedFiles: string[] = [];

  for (const route of DASHBOARD_STAGING_ROUTES) {
    const handler = await loadRouteHandler(route.modulePath);
    const response = await handler(requestForLeague(route.apiPath, league));
    const payload = await response.json();
    await writeJson(path.join(leagueDir, route.stagingFileName), sanitizePublicPayload(route.stagingFileName, payload, league));
    generatedFiles.push(route.stagingFileName);

    if (route.supportsExperiments) {
      for (const experiment of listPerformanceReplayExperiments()) {
        const experimentResponse = await handler(
          requestForLeague(route.apiPath, league, { experiment: experiment.id })
        );
        const experimentPayload = await experimentResponse.json();
        const experimentFileName = buildPerformanceExperimentStagingFileName(experiment.id);
        await writeJson(
          path.join(leagueDir, experimentFileName),
          sanitizePublicPayload(experimentFileName, experimentPayload, league)
        );
        generatedFiles.push(experimentFileName);
      }
    }
  }

  await writeJson(path.join(leagueDir, "meta.json"), {
    generated_at_utc: generatedAtUtc,
    league,
    primary_league: PRIMARY_STAGING_LEAGUE,
    shipping_role: league === PRIMARY_STAGING_LEAGUE ? "primary" : "supported",
    mode: "static-staging-snapshot",
    note: "Generated locally from the current dashboard data sources for GitHub Pages staging.",
    files: generatedFiles,
  });
}

async function main(): Promise<void> {
  for (const league of SHIPPED_STAGING_LEAGUES) {
    await generateLeagueSnapshot(league);
  }
  // Maintainer note: Pages publishes these committed artifacts directly.
  // Regenerating without committing leaves the local dashboard and staging out of sync.
  await writeJson(path.join(outputRoot, "manifest.json"), buildStagingManifestPayload(generatedAtUtc));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
