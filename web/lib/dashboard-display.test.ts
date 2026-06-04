import assert from "node:assert/strict";
import test from "node:test";

import { describeFreshForecastStatus, formatCentralTimestamp, shortenToken } from "@/lib/dashboard-display";

test("describeFreshForecastStatus explains a no-slate day", () => {
  const status = describeFreshForecastStatus({
    status: "no_slate",
    scheduledGameCount: 0,
    dateCentral: "2026-06-04",
  });

  assert.equal(status.headline, "No MLB slate for Jun 4, 2026");
  assert.match(status.detail, /intentionally empty/);
});

test("describeFreshForecastStatus explains missing forecasts with scheduled games", () => {
  const status = describeFreshForecastStatus({
    status: "missing",
    scheduledGameCount: 12,
    dateCentral: "2026-06-04",
  });

  assert.equal(status.headline, "Forecast snapshot missing");
  assert.match(status.detail, /12 MLB games found/);
});

test("formatCentralTimestamp converts UTC timestamps to Central labels", () => {
  assert.equal(formatCentralTimestamp("2026-06-04T18:30:00Z"), "Jun 4, 2026 1:30 PM CT");
});

test("shortenToken truncates long identifiers while preserving short labels", () => {
  assert.equal(shortenToken("abc123", 8), "abc123");
  assert.equal(shortenToken("abcdefghijklmnopqrstuvwxyz", 8), "abcdefgh...");
  assert.equal(shortenToken("", 8), "-");
});
