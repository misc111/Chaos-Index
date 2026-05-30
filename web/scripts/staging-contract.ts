import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { DASHBOARD_STAGING_ROUTES } from "../lib/generated/dashboard-routes";
import { ALL_LEAGUES, type LeagueCode } from "../lib/generated/league-registry";
import { centralDateKeyFromTimestamp, normalizeCentralDateKey, shiftCentralDateKey } from "../lib/games-today";
import {
  buildPerformanceExperimentStagingFileName,
  listPerformanceReplayExperiments,
} from "../lib/performance-replay-experiments";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(scriptDir, "..");

export const PRIMARY_STAGING_LEAGUE: LeagueCode = "MLB";
export const SHIPPED_STAGING_LEAGUES: readonly LeagueCode[] = [PRIMARY_STAGING_LEAGUE];
export const LEGACY_STAGING_LEAGUES: readonly LeagueCode[] = ALL_LEAGUES.filter(
  (league) => league !== PRIMARY_STAGING_LEAGUE
);

export const DEFAULT_STAGING_OUTPUT_ROOT = path.join(appRoot, "public", "staging-data");
export const DEFAULT_STAGING_MANIFEST_PATH = path.join(DEFAULT_STAGING_OUTPUT_ROOT, "manifest.json");

function isLeagueCode(value: unknown): value is LeagueCode {
  return typeof value === "string" && (ALL_LEAGUES as readonly string[]).includes(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function validateEvidenceStatusPayload(args: {
  league: LeagueCode;
  fileName: string;
  payload: Record<string, unknown>;
}): string[] {
  const { league, fileName, payload } = args;
  const errors: string[] = [];
  const currentBest = asRecord(payload.current_best);
  const source = fileName === "nested-tournament.json" ? currentBest : payload;
  if (fileName !== "nested-tournament.json" && fileName !== "research-desk.json") {
    return errors;
  }

  if (fileName === "nested-tournament.json" && source.latest_artifact_role !== "latest_research_recommendation") {
    errors.push(`${league} ${fileName} must expose nested tournament evidence as latest_research_recommendation.`);
  }
  if (fileName === "nested-tournament.json" && source.promotion_eligible !== false) {
    errors.push(`${league} ${fileName} must block nested tournament evidence from promotion.`);
  }
  if (!source.latest_artifact_role) {
    errors.push(`${league} ${fileName} must declare latest_artifact_role.`);
  }
  if (source.promotion_eligible !== true && source.promotion_eligible !== false) {
    errors.push(`${league} ${fileName} must declare boolean promotion_eligible.`);
  }
  const evidenceStatus = asRecord(source.evidence_status);
  const evidenceStage = String(evidenceStatus.evidence_stage || source.evidence_stage || "");
  const allowedStages = new Set(["fixture_demo_smoke", "promotion_eligible", "production_ready", "research_only"]);
  if (!allowedStages.has(evidenceStage)) {
    errors.push(`${league} ${fileName} must expose a valid evidence_stage.`);
  }
  if (!evidenceStatus.pointer_semantics) {
    errors.push(`${league} ${fileName} must expose evidence_status.pointer_semantics.`);
  }
  if (source.latest_artifact_role === "latest_research_recommendation" && source.promotion_eligible === true) {
    errors.push(`${league} ${fileName} cannot mark a latest research recommendation as promotion eligible.`);
  }
  if (
    asRecord(evidenceStatus).fixture_demo_smoke === true &&
    (source.promotion_eligible === true || asRecord(evidenceStatus).promotion_eligible === true)
  ) {
    errors.push(`${league} ${fileName} cannot mark fixture/demo/smoke evidence as promotion eligible.`);
  }
  if (
    (source.promotion_eligible === true || evidenceStatus.promotion_eligible === true) &&
    (evidenceStatus.production_grade !== true || evidenceStatus.full_immutable_pregame_ledger !== true)
  ) {
    errors.push(`${league} ${fileName} promotion-eligible evidence must be production-grade full-ledger evidence.`);
  }
  if (
    (source.production_ready === true || evidenceStatus.production_ready === true) &&
    (source.promotion_eligible !== true || evidenceStatus.promotion_gate_passed !== true)
  ) {
    errors.push(`${league} ${fileName} production-ready evidence must pass promotion gates.`);
  }
  return errors;
}

function validateForecastFreshnessPayload(args: {
  league: LeagueCode;
  fileName: string;
  payload: Record<string, unknown>;
}): string[] {
  const { league, fileName, payload } = args;
  const errors: string[] = [];
  if (fileName !== "games-today.json" && fileName !== "market-board.json" && fileName !== "research-desk.json") {
    return [];
  }

  const dateCentral = normalizeCentralDateKey(String(payload.date_central || ""));
  const forecastAsOfCentral = centralDateKeyFromTimestamp(
    typeof payload.as_of_utc === "string" ? payload.as_of_utc : null
  );
  const minimumForecastDate = dateCentral ? shiftCentralDateKey(dateCentral, -1) : null;
  if (dateCentral && forecastAsOfCentral && minimumForecastDate && forecastAsOfCentral < minimumForecastDate) {
    errors.push(
      `${league} ${fileName} forecast as_of_utc central date ${forecastAsOfCentral} is stale for date_central ${dateCentral}.`,
    );
  }
  const scheduledGameCount = Number(payload.scheduled_game_count);
  const freshForecastStatus = String(payload.fresh_forecast_status || "");
  const allowedFreshForecastStatuses = new Set(["available", "missing", "no_slate"]);
  if (!Number.isInteger(scheduledGameCount) || scheduledGameCount < 0) {
    errors.push(`${league} ${fileName} must expose nonnegative scheduled_game_count.`);
  }
  if (!allowedFreshForecastStatuses.has(freshForecastStatus)) {
    errors.push(`${league} ${fileName} must expose a valid fresh_forecast_status.`);
  }
  if (payload.as_of_utc && freshForecastStatus !== "available") {
    errors.push(`${league} ${fileName} fresh_forecast_status must be available when as_of_utc is present.`);
  }
  if (!payload.as_of_utc && scheduledGameCount > 0 && freshForecastStatus !== "missing") {
    errors.push(`${league} ${fileName} fresh_forecast_status must be missing when a scheduled slate has no as_of_utc.`);
  }
  return errors;
}

export function listRequiredStagingFiles(): string[] {
  const requiredFiles = new Set<string>();

  for (const route of DASHBOARD_STAGING_ROUTES) {
    requiredFiles.add(route.stagingFileName);
    if (route.supportsExperiments) {
      for (const experiment of listPerformanceReplayExperiments()) {
        requiredFiles.add(buildPerformanceExperimentStagingFileName(experiment.id));
      }
    }
  }

  return [...requiredFiles].sort();
}

export type StagingManifest = {
  contract: "mlb-first-static-staging";
  generated_at_utc: string;
  primary_league: LeagueCode;
  shipped_leagues: LeagueCode[];
  leagues: LeagueCode[];
  required_files_by_league: Record<LeagueCode, string[]>;
};

export function buildStagingManifestPayload(generatedAtUtc: string): StagingManifest {
  const requiredFiles = listRequiredStagingFiles();
  return {
    contract: "mlb-first-static-staging",
    generated_at_utc: generatedAtUtc,
    primary_league: PRIMARY_STAGING_LEAGUE,
    shipped_leagues: [...SHIPPED_STAGING_LEAGUES],
    leagues: [...SHIPPED_STAGING_LEAGUES],
    required_files_by_league: Object.fromEntries(
      SHIPPED_STAGING_LEAGUES.map((league) => [league, [...requiredFiles]])
    ) as Record<LeagueCode, string[]>,
  };
}

export function validateStagingManifest(manifest: unknown): string[] {
  const errors: string[] = [];
  const raw = (manifest || {}) as Record<string, unknown>;
  const shippedLeagues = Array.isArray(raw.shipped_leagues) ? raw.shipped_leagues : [];
  const leaguesAlias = Array.isArray(raw.leagues) ? raw.leagues : [];
  const requiredFilesByLeague =
    raw.required_files_by_league && typeof raw.required_files_by_league === "object"
      ? (raw.required_files_by_league as Record<string, unknown>)
      : {};
  const expectedFiles = listRequiredStagingFiles();

  if (raw.contract !== "mlb-first-static-staging") {
    errors.push("manifest.contract must be `mlb-first-static-staging`.");
  }
  if (raw.primary_league !== PRIMARY_STAGING_LEAGUE) {
    errors.push(`manifest.primary_league must be ${PRIMARY_STAGING_LEAGUE}.`);
  }
  if (!Array.isArray(raw.shipped_leagues)) {
    errors.push("manifest.shipped_leagues must be an array.");
  }
  if (!Array.isArray(raw.leagues)) {
    errors.push("manifest.leagues must be an array.");
  }

  if (Array.isArray(shippedLeagues)) {
    if (!shippedLeagues.every(isLeagueCode)) {
      errors.push("manifest.shipped_leagues must contain only known league codes.");
    }
    if (shippedLeagues[0] !== PRIMARY_STAGING_LEAGUE) {
      errors.push(`manifest.shipped_leagues must list ${PRIMARY_STAGING_LEAGUE} first.`);
    }
    if (!shippedLeagues.includes(PRIMARY_STAGING_LEAGUE)) {
      errors.push(`manifest.shipped_leagues must include ${PRIMARY_STAGING_LEAGUE}.`);
    }
  }

  if (Array.isArray(leaguesAlias)) {
    if (!leaguesAlias.every(isLeagueCode)) {
      errors.push("manifest.leagues must contain only known league codes.");
    }
    if (JSON.stringify(leaguesAlias) !== JSON.stringify(shippedLeagues)) {
      errors.push("manifest.leagues must match manifest.shipped_leagues exactly.");
    }
  }

  for (const league of SHIPPED_STAGING_LEAGUES) {
    const files = requiredFilesByLeague[league];
    if (!Array.isArray(files)) {
      errors.push(`manifest.required_files_by_league.${league} must be an array.`);
      continue;
    }
    const normalized = files.filter((value): value is string => typeof value === "string").sort();
    if (JSON.stringify(normalized) !== JSON.stringify(expectedFiles)) {
      errors.push(`manifest.required_files_by_league.${league} must match the generated staging route file set.`);
    }
  }

  return errors;
}

export async function collectCommittedStagingSnapshotErrors(
  outputRoot: string = DEFAULT_STAGING_OUTPUT_ROOT
): Promise<string[]> {
  const errors: string[] = [];
  const manifestPath = path.join(outputRoot, "manifest.json");
  let manifest: StagingManifest | null = null;

  try {
    manifest = JSON.parse(await fs.readFile(manifestPath, "utf8")) as StagingManifest;
  } catch (error) {
    errors.push(
      error instanceof Error
        ? `unable to read staging manifest at ${manifestPath}: ${error.message}`
        : `unable to read staging manifest at ${manifestPath}`
    );
    return errors;
  }

  errors.push(...validateStagingManifest(manifest));
  if (errors.length > 0) {
    return errors;
  }

  for (const league of manifest.shipped_leagues) {
    const leagueDir = path.join(outputRoot, league.toLowerCase());
    const metaPath = path.join(leagueDir, "meta.json");
    const requiredFiles = manifest.required_files_by_league[league] || [];

    try {
      await fs.access(leagueDir);
    } catch {
      errors.push(`missing staging directory for ${league}: ${leagueDir}`);
      continue;
    }

    let meta: Record<string, unknown> = {};
    try {
      meta = JSON.parse(await fs.readFile(metaPath, "utf8")) as Record<string, unknown>;
    } catch (error) {
      errors.push(
        error instanceof Error
          ? `unable to read ${league} meta.json: ${error.message}`
          : `unable to read ${league} meta.json`
      );
      continue;
    }

    if (meta.league !== league) {
      errors.push(`${league} meta.json must declare league ${league}.`);
    }
    if (meta.primary_league !== manifest.primary_league) {
      errors.push(`${league} meta.json must declare primary_league ${manifest.primary_league}.`);
    }
    const expectedRole = league === manifest.primary_league ? "primary" : "supported";
    if (meta.shipping_role !== expectedRole) {
      errors.push(`${league} meta.json must declare shipping_role ${expectedRole}.`);
    }

    const metaFiles = Array.isArray(meta.files)
      ? meta.files.filter((value): value is string => typeof value === "string")
      : [];
    const allowedFiles = new Set([...requiredFiles, "meta.json", ".gitkeep"]);
    const entries = await fs.readdir(leagueDir, { withFileTypes: true });

    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      if (!allowedFiles.has(entry.name)) {
        errors.push(`unexpected ${league} staging payload ${entry.name}`);
      }
    }

    for (const fileName of requiredFiles) {
      const payloadPath = path.join(leagueDir, fileName);
      try {
        await fs.access(payloadPath);
      } catch {
        errors.push(`missing ${league} staging payload ${fileName}`);
      }
      if (!metaFiles.includes(fileName)) {
        errors.push(`${league} meta.json must list ${fileName}.`);
      }
      if (fileName === "nested-tournament.json" || fileName === "research-desk.json") {
        try {
          const payload = JSON.parse(await fs.readFile(payloadPath, "utf8")) as Record<string, unknown>;
          errors.push(...validateEvidenceStatusPayload({ league, fileName, payload }));
        } catch (error) {
          errors.push(
            error instanceof Error
              ? `unable to read ${league} ${fileName}: ${error.message}`
              : `unable to read ${league} ${fileName}`
          );
        }
      }
      if (fileName === "games-today.json" || fileName === "market-board.json" || fileName === "research-desk.json") {
        try {
          const payload = JSON.parse(await fs.readFile(payloadPath, "utf8")) as Record<string, unknown>;
          errors.push(...validateForecastFreshnessPayload({ league, fileName, payload }));
        } catch (error) {
          errors.push(
            error instanceof Error
              ? `unable to read ${league} ${fileName}: ${error.message}`
              : `unable to read ${league} ${fileName}`
          );
        }
      }
    }
  }

  return errors;
}
