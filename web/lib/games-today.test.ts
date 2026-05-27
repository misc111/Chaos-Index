import assert from "node:assert/strict";
import test from "node:test";

import { centralTodayDateKey } from "./games-today";

test("centralTodayDateKey honors the dashboard date override for static staging", () => {
  const previous = process.env.DASHBOARD_DATE_CENTRAL;
  process.env.DASHBOARD_DATE_CENTRAL = "2026-05-04";

  try {
    assert.equal(centralTodayDateKey(), "2026-05-04");
  } finally {
    if (previous === undefined) {
      delete process.env.DASHBOARD_DATE_CENTRAL;
    } else {
      process.env.DASHBOARD_DATE_CENTRAL = previous;
    }
  }
});

test("centralTodayDateKey ignores malformed dashboard date overrides", () => {
  const previous = process.env.DASHBOARD_DATE_CENTRAL;
  process.env.DASHBOARD_DATE_CENTRAL = "not-a-date";

  try {
    assert.match(centralTodayDateKey(), /^\d{4}-\d{2}-\d{2}$/);
  } finally {
    if (previous === undefined) {
      delete process.env.DASHBOARD_DATE_CENTRAL;
    } else {
      process.env.DASHBOARD_DATE_CENTRAL = previous;
    }
  }
});
