import assert from "node:assert/strict";
import test from "node:test";

import { buildDashboardDataUrl, fetchDashboardJson, isStaticStagingBuild } from "./static-staging";

const originalStaticStaging = process.env.NEXT_PUBLIC_STATIC_STAGING;

function setStaticStaging(value: "1" | undefined): void {
  if (value === undefined) {
    delete process.env.NEXT_PUBLIC_STATIC_STAGING;
    return;
  }
  process.env.NEXT_PUBLIC_STATIC_STAGING = value;
}

test.afterEach(() => {
  setStaticStaging(originalStaticStaging === "1" ? "1" : undefined);
});

test("uses the live API route unless static staging is explicitly enabled", () => {
  setStaticStaging(undefined);

  assert.equal(isStaticStagingBuild(), false);
  assert.equal(buildDashboardDataUrl("gamesToday", "/api/games-today", "MLB"), "/api/games-today");
});

test("uses committed staging snapshots when static staging is explicitly enabled", () => {
  setStaticStaging("1");

  assert.equal(isStaticStagingBuild(), true);
  assert.equal(buildDashboardDataUrl("gamesToday", "/api/games-today", "MLB"), "/staging-data/mlb/games-today.json");
});

test("fetches live routes without static snapshot caching in normal dashboard runtime", async () => {
  setStaticStaging(undefined);

  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; cache?: RequestCache }> = [];

  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), cache: init?.cache });
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  try {
    const payload = await fetchDashboardJson<{ ok: boolean }>("gamesToday", "/api/games-today", "MLB");

    assert.deepEqual(payload, { ok: true });
    assert.deepEqual(calls, [{ url: "/api/games-today", cache: "no-store" }]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

