import assert from "node:assert/strict";
import test from "node:test";

import { DASHBOARD_ROUTES, DASHBOARD_STAGING_ROUTES } from "../lib/generated/dashboard-routes";

test("research-admin stays out of staged snapshots while research-desk is included", () => {
  const stagedKeys = new Set(DASHBOARD_STAGING_ROUTES.map((route) => route.key));
  const routeKeys = new Set(DASHBOARD_ROUTES.map((route) => route.key));

  assert.equal(routeKeys.has("researchAdmin"), true);
  assert.equal(routeKeys.has("researchDesk"), true);
  assert.equal(stagedKeys.has("researchDesk"), true);
  assert.equal(stagedKeys.has("researchAdmin"), false);
});
