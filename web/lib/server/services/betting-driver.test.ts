import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { execFileSync } from "node:child_process";

import {
  getActiveChampionSummary,
  getBettingDriverContext,
  getPreferredBettingModelName,
} from "./betting-driver";

test("NBA prefers glm_elastic_net as the betting driver", () => {
  assert.equal(getPreferredBettingModelName("NBA"), "glm_elastic_net");
  assert.equal(getPreferredBettingModelName("NBA", "historicalReplay"), "glm_elastic_net");
});

test("other leagues keep the ensemble betting driver", () => {
  assert.equal(getPreferredBettingModelName("NHL"), "ensemble");
  assert.equal(getPreferredBettingModelName("NCAAM"), "ensemble");
});

test("NBA prefers the active champion when one exists", () => {
  const tempDir = mkdtempSync(path.join(tmpdir(), "sportsmodeling-driver-"));
  const dbPath = path.join(tempDir, "test.sqlite");
  const previousDbPath = process.env.SPORTS_DB_PATH;
  process.env.SPORTS_DB_PATH = dbPath;
  try {
    execFileSync("sqlite3", [
      dbPath,
      `
      CREATE TABLE active_champions (
        league TEXT NOT NULL,
        profile_key TEXT NOT NULL DEFAULT 'default',
        model_name TEXT NOT NULL,
        source_run_id TEXT,
        source_brief_id TEXT,
        promoted_at_utc TEXT NOT NULL,
        descriptor_json TEXT NOT NULL,
        policy_json TEXT,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        PRIMARY KEY (league, profile_key)
      );
      INSERT INTO active_champions(
        league, profile_key, model_name, source_run_id, source_brief_id,
        promoted_at_utc, descriptor_json, policy_json, created_at_utc, updated_at_utc
      ) VALUES(
        'NBA', 'default', 'glm_ridge', 'run-1', 'brief-1',
        '2026-03-27T00:00:00Z', '{"model_name":"glm_ridge"}', '{"promoted":true}',
        '2026-03-27T00:00:00Z', '2026-03-27T00:00:00Z'
      );
      `,
    ]);

    assert.equal(getPreferredBettingModelName("NBA"), "glm_ridge");
    assert.equal(getActiveChampionSummary("NBA")?.source_run_id, "run-1");
  } finally {
    if (previousDbPath === undefined) {
      delete process.env.SPORTS_DB_PATH;
    } else {
      process.env.SPORTS_DB_PATH = previousDbPath;
    }
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("betting driver context falls back to the NBA default model when no champion is active", () => {
  const tempDir = mkdtempSync(path.join(tmpdir(), "sportsmodeling-driver-"));
  const dbPath = path.join(tempDir, "test.sqlite");
  const previousDbPath = process.env.SPORTS_DB_PATH;
  process.env.SPORTS_DB_PATH = dbPath;
  try {
    const context = getBettingDriverContext("NBA");
    assert.equal(context.preferred_model_name, "glm_elastic_net");
    assert.equal(context.desk_posture, "normal");
    assert.equal(context.champion, null);
  } finally {
    if (previousDbPath === undefined) {
      delete process.env.SPORTS_DB_PATH;
    } else {
      process.env.SPORTS_DB_PATH = previousDbPath;
    }
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("betting driver context reuses the shared guarded posture and champion summary", () => {
  const tempDir = mkdtempSync(path.join(tmpdir(), "sportsmodeling-driver-"));
  const dbPath = path.join(tempDir, "test.sqlite");
  const previousDbPath = process.env.SPORTS_DB_PATH;
  process.env.SPORTS_DB_PATH = dbPath;
  try {
    execFileSync("sqlite3", [
      dbPath,
      `
      CREATE TABLE active_champions (
        league TEXT NOT NULL,
        profile_key TEXT NOT NULL DEFAULT 'default',
        model_name TEXT NOT NULL,
        source_run_id TEXT,
        source_brief_id TEXT,
        promoted_at_utc TEXT NOT NULL,
        descriptor_json TEXT NOT NULL,
        policy_json TEXT,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        PRIMARY KEY (league, profile_key)
      );
      CREATE TABLE change_points (
        change_point_id INTEGER PRIMARY KEY,
        as_of_utc TEXT NOT NULL,
        model_name TEXT NOT NULL,
        details_json TEXT NOT NULL
      );
      INSERT INTO active_champions(
        league, profile_key, model_name, source_run_id, source_brief_id,
        promoted_at_utc, descriptor_json, policy_json, created_at_utc, updated_at_utc
      ) VALUES(
        'NBA', 'default', 'glm_ridge', 'run-7', 'brief-2',
        '2026-03-27T00:00:00Z', '{"model_name":"glm_ridge"}', '{"promoted":true}',
        '2026-03-27T00:00:00Z', '2026-03-27T00:00:00Z'
      );
      INSERT INTO change_points(change_point_id, as_of_utc, model_name, details_json) VALUES
        (1, '2026-03-27T09:00:00Z', 'ensemble', '{"date":"2026-03-27"}'),
        (2, '2026-03-27T09:00:00Z', 'glm_elastic_net', '{"date":"2026-03-26"}'),
        (3, '2026-03-27T09:00:00Z', 'glm_ridge', '{"date":"2026-03-25"}');
      `,
    ]);

    const context = getBettingDriverContext("NBA");
    assert.equal(context.preferred_model_name, "glm_ridge");
    assert.equal(context.desk_posture, "guarded");
    assert.equal(context.champion?.source_run_id, "run-7");
    assert.equal(context.champion?.policy?.promoted, true);
  } finally {
    if (previousDbPath === undefined) {
      delete process.env.SPORTS_DB_PATH;
    } else {
      process.env.SPORTS_DB_PATH = previousDbPath;
    }
    rmSync(tempDir, { recursive: true, force: true });
  }
});
