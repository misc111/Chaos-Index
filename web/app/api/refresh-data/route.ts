import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { type LeagueCode, leagueFromRequest } from "@/lib/league";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RefreshState = {
  running: boolean;
  league?: LeagueCode;
  startedAtUtc?: string;
};

declare global {
  var __sportsModelingRefreshState: RefreshState | undefined;
}

const refreshState: RefreshState = globalThis.__sportsModelingRefreshState || { running: false };
globalThis.__sportsModelingRefreshState = refreshState;

const MAX_LOG_CHARS = 16000;

function appendChunk(current: string, chunk: string): string {
  const next = current + chunk;
  if (next.length <= MAX_LOG_CHARS) {
    return next;
  }
  return next.slice(next.length - MAX_LOG_CHARS);
}

function repoRootPath(): string {
  return path.resolve(process.cwd(), "..");
}

function configPathForLeague(league: LeagueCode): string {
  const envOverride = league === "NBA" ? process.env.NBA_CONFIG_PATH : process.env.NHL_CONFIG_PATH;
  const fallback = league === "NBA" ? "configs/nba.yaml" : "configs/nhl.yaml";
  return path.resolve(repoRootPath(), envOverride || fallback);
}

function runDataRefreshPipeline(configPath: string): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", ["-m", "src.cli", "refresh-data", "--config", configPath], {
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

function trimLog(log: string): string {
  return log.trim().slice(-6000);
}

export async function POST(request: Request) {
  const league = leagueFromRequest(request);

  if (refreshState.running) {
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `A refresh is already running for ${refreshState.league || "another league"}.`,
        started_at_utc: refreshState.startedAtUtc,
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

  refreshState.running = true;
  refreshState.league = league;
  refreshState.startedAtUtc = new Date().toISOString();

  try {
    const runResult = await runDataRefreshPipeline(configPath);
    if (runResult.code !== 0) {
      return NextResponse.json(
        {
          ok: false,
          league,
          error: "Data refresh failed.",
          details: trimLog(runResult.stderr || runResult.stdout || `Command exited with code ${runResult.code}`),
        },
        { status: 500 }
      );
    }

    return NextResponse.json({
      ok: true,
      league,
      started_at_utc: refreshState.startedAtUtc,
      refreshed_at_utc: new Date().toISOString(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        ok: false,
        league,
        error: `Data refresh failed: ${message}`,
      },
      { status: 500 }
    );
  } finally {
    refreshState.running = false;
    refreshState.league = undefined;
    refreshState.startedAtUtc = undefined;
  }
}
