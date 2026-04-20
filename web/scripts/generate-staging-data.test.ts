import assert from "node:assert/strict";
import test from "node:test";

import { DASHBOARD_ROUTES, DASHBOARD_STAGING_ROUTES } from "../lib/generated/dashboard-routes";
import {
  buildStagingManifestPayload,
  listRequiredStagingFiles,
  PRIMARY_STAGING_LEAGUE,
  SHIPPED_STAGING_LEAGUES,
} from "./staging-contract";

test("research-admin stays out of staged snapshots while research-desk is included", () => {
  const stagedKeys = new Set(DASHBOARD_STAGING_ROUTES.map((route) => route.key));
  const routeKeys = new Set(DASHBOARD_ROUTES.map((route) => route.key));

  assert.equal(routeKeys.has("researchAdmin"), true);
  assert.equal(routeKeys.has("researchDesk"), true);
  assert.equal(stagedKeys.has("researchDesk"), true);
  assert.equal(stagedKeys.has("researchAdmin"), false);
});

test("staging manifest declares MLB as the primary shipped lane", () => {
  const manifest = buildStagingManifestPayload("2026-04-20T00:00:00.000Z");
  const requiredFiles = listRequiredStagingFiles();

  assert.equal(manifest.primary_league, PRIMARY_STAGING_LEAGUE);
  assert.deepEqual(manifest.shipped_leagues, SHIPPED_STAGING_LEAGUES);
  assert.deepEqual(manifest.leagues, SHIPPED_STAGING_LEAGUES);
  assert.deepEqual(manifest.required_files_by_league.MLB, requiredFiles);
  assert.deepEqual(manifest.required_files_by_league.NHL, requiredFiles);
  assert.deepEqual(manifest.required_files_by_league.NBA, requiredFiles);
});
