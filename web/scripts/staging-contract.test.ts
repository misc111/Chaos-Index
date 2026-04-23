import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { fileURLToPath } from "node:url";

import {
  buildStagingManifestPayload,
  collectCommittedStagingSnapshotErrors,
  DEFAULT_STAGING_MANIFEST_PATH,
  listRequiredStagingFiles,
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
  assert.deepEqual(manifest.shipped_leagues, [PRIMARY_STAGING_LEAGUE]);
  assert.deepEqual(manifest.leagues, [PRIMARY_STAGING_LEAGUE]);
});

test("staging snapshot verifier rejects unexpected shipped league files", async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "staging-contract-"));
  const leagueDir = path.join(tempRoot, "mlb");
  const requiredFiles = listRequiredStagingFiles();

  try {
    await fs.mkdir(leagueDir, { recursive: true });
    await fs.writeFile(path.join(tempRoot, "manifest.json"), JSON.stringify(buildStagingManifestPayload("2026-04-23T00:00:00.000Z")));
    await fs.writeFile(path.join(leagueDir, ".gitkeep"), "");
    for (const fileName of requiredFiles) {
      await fs.writeFile(path.join(leagueDir, fileName), "{}");
    }
    await fs.writeFile(
      path.join(leagueDir, "meta.json"),
      JSON.stringify({
        league: PRIMARY_STAGING_LEAGUE,
        primary_league: PRIMARY_STAGING_LEAGUE,
        shipping_role: "primary",
        files: requiredFiles,
      })
    );
    await fs.writeFile(path.join(leagueDir, "stale-route.json"), "{}");

    const errors = await collectCommittedStagingSnapshotErrors(tempRoot);

    assert.ok(errors.includes("unexpected MLB staging payload stale-route.json"));
  } finally {
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
});

test("staging root contains only the MLB payload directory", async () => {
  const stagingRoot = path.join(repoRoot, "web", "public", "staging-data");
  const rootEntries = (await fs.readdir(stagingRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  assert.deepEqual(rootEntries, ["mlb"]);
});

test("pages workflow verifies the staging contract instead of hardcoded league files", async () => {
  const workflowPath = path.join(repoRoot, ".github", "workflows", "pages-staging.yml");
  const workflow = await fs.readFile(workflowPath, "utf8");

  assert.match(workflow, /node --import tsx scripts\/verify-staging-contract\.ts/);
});

test("pages build script runs the shared staging contract verifier before Next build", async () => {
  const buildScriptPath = path.join(repoRoot, "web", "scripts", "build-pages.mjs");
  const buildScript = await fs.readFile(buildScriptPath, "utf8");

  assert.match(buildScript, /scripts\/verify-staging-contract\.ts/);
  assert.match(buildScript, /process\.execPath/);
});
