import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import { fileURLToPath } from "node:url";

import {
  collectCommittedStagingSnapshotErrors,
  DEFAULT_STAGING_MANIFEST_PATH,
  PRIMARY_STAGING_LEAGUE,
} from "./staging-contract";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..", "..");

test("committed staging snapshot satisfies the MLB-first shipping contract", async () => {
  const errors = await collectCommittedStagingSnapshotErrors();
  assert.deepEqual(errors, []);
});

test("committed staging manifest makes MLB the primary shipped league", async () => {
  const manifest = JSON.parse(await fs.readFile(DEFAULT_STAGING_MANIFEST_PATH, "utf8")) as {
    primary_league: string;
    shipped_leagues: string[];
  };

  assert.equal(manifest.primary_league, PRIMARY_STAGING_LEAGUE);
  assert.equal(manifest.shipped_leagues[0], PRIMARY_STAGING_LEAGUE);
  assert.ok(manifest.shipped_leagues.includes(PRIMARY_STAGING_LEAGUE));
});

test("pages workflow verifies the staging contract instead of hardcoded legacy league files", async () => {
  const workflowPath = path.join(repoRoot, ".github", "workflows", "pages-staging.yml");
  const workflow = await fs.readFile(workflowPath, "utf8");

  assert.match(workflow, /node --import tsx scripts\/verify-staging-contract\.ts/);
  assert.doesNotMatch(workflow, /staging-data\/nhl\/predictions\.json/);
  assert.doesNotMatch(workflow, /staging-data\/nba\/predictions\.json/);
});

test("pages build script runs the shared staging contract verifier before Next build", async () => {
  const buildScriptPath = path.join(repoRoot, "web", "scripts", "build-pages.mjs");
  const buildScript = await fs.readFile(buildScriptPath, "utf8");

  assert.match(buildScript, /scripts\/verify-staging-contract\.ts/);
  assert.match(buildScript, /process\.execPath/);
});
