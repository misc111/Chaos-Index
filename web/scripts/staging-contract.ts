import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { DASHBOARD_STAGING_ROUTES } from "../lib/generated/dashboard-routes";
import { ALL_LEAGUES, type LeagueCode } from "../lib/generated/league-registry";
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
      try {
        await fs.access(path.join(leagueDir, fileName));
      } catch {
        errors.push(`missing ${league} staging payload ${fileName}`);
      }
      if (!metaFiles.includes(fileName)) {
        errors.push(`${league} meta.json must list ${fileName}.`);
      }
    }
  }

  return errors;
}
