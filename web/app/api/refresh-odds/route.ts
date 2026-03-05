import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { runSqlJson } from "@/lib/db";
import { type LeagueCode, leagueFromRequest } from "@/lib/league";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RefreshOddsState = {
  running: boolean;
  league?: LeagueCode;
  startedAtUtc?: string;
};

declare global {
  // eslint-disable-next-line no-var
  var __sportsModelingRefreshOddsState: RefreshOddsState | undefined;
}

const refreshOddsState: RefreshOddsState = globalThis.__sportsModelingRefreshOddsState || { running: false };
globalThis.__sportsModelingRefreshOddsState = refreshOddsState;

const MAX_LOG_CHARS = 16000;

function appendChunk(current: string, chunk: string): string {
  const next = current + chunk;
  if (next.length <= MAX_LOG_CHARS) {
    return next;
  }
  return next.slice(next.length - MAX_LOG_CHARS);
}

function trimLog(log: string): string {
  return log.trim().slice(-6000);
}

function repoRootPath(): string {
  return path.resolve(process.cwd(), "..");
}

function configPathForLeague(league: LeagueCode): string {
  const envOverride = league === "NBA" ? process.env.NBA_CONFIG_PATH : process.env.NHL_CONFIG_PATH;
  const fallback = league === "NBA" ? "configs/nba.yaml" : "configs/nhl.yaml";
  return path.resolve(repoRootPath(), envOverride || fallback);
}

function runOddsRefresh(configPath: string): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", ["-m", "src.cli", "fetch-odds", "--config", configPath], {
      cwd: repoRootPath(),
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk: Buffer | string) => {
      stdout = appendChunk(stdout, chunk.toString());
    });

    child.stderr.on("data", (chunk: Buffer | string) => {
      stderr = appendChunk(stderr, chunk.toString());
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}

export async function POST(request: Request) {
  const league = leagueFromRequest(request);

  if (refreshOddsState.running) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `An odds refresh is already running for ${refreshOddsState.league || "another league"}.`,
        started_at_utc: refreshOddsState.startedAtUtc,
      },
      { status: 409 }
    );
  }

  const configPath = configPathForLeague(league);
  if (!fs.existsSync(configPath)) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `Config file not found: ${configPath}`,
      },
      { status: 500 }
    );
  }

  refreshOddsState.running = true;
  refreshOddsState.league = league;
  refreshOddsState.startedAtUtc = new Date().toISOString();

  try {
    const runResult = await runOddsRefresh(configPath);
    if (runResult.code !== 0) {
      return NextResponse.json(
        {
          ok: false,
          league,
          error: "Odds refresh failed.",
          details: trimLog(runResult.stderr || runResult.stdout || `Command exited with code ${runResult.code}`),
        },
        { status: 500 }
      );
    }

    const latestOddsSnapshot = runSqlJson(
      `
      SELECT odds_snapshot_id, as_of_utc, event_count, row_count
      FROM odds_snapshots
      WHERE league = '${league}'
      ORDER BY DATETIME(as_of_utc) DESC
      LIMIT 1
      `,
      { league }
    );

    return NextResponse.json({
      ok: true,
      league,
      started_at_utc: refreshOddsState.startedAtUtc,
      refreshed_at_utc: new Date().toISOString(),
      odds_snapshot_id: latestOddsSnapshot?.[0]?.odds_snapshot_id || null,
      odds_as_of_utc: latestOddsSnapshot?.[0]?.as_of_utc || null,
      event_count: latestOddsSnapshot?.[0]?.event_count ?? null,
      row_count: latestOddsSnapshot?.[0]?.row_count ?? null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `Odds refresh failed: ${message}`,
      },
      { status: 500 }
    );
  } finally {
    refreshOddsState.running = false;
    refreshOddsState.league = undefined;
    refreshOddsState.startedAtUtc = undefined;
  }
}
