import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { type LeagueCode, leagueFromRequest } from "@/lib/league";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ProgressEvent = {
  id: number;
  ts_utc: string;
  kind: string;
  stage: string;
  status: string;
  message?: string;
  model?: string;
  phase?: string;
  fold?: number;
  fold_total?: number;
  model_total?: number;
  selected_models?: string[];
};

const ALL_MODEL_NAMES = [
  "elo_baseline",
  "dynamic_rating",
  "glm_logit",
  "gbdt",
  "rf",
  "two_stage",
  "goals_poisson",
  "simulation_first",
  "bayes_bt_state_space",
  "bayes_goals",
  "nn_mlp",
] as const;

type ModelName = (typeof ALL_MODEL_NAMES)[number];

type TrainingState = {
  running: boolean;
  league?: LeagueCode;
  requestedModels?: ModelName[];
  startedAtUtc?: string;
  finishedAtUtc?: string;
  exitCode?: number | null;
  error?: string;
  events: ProgressEvent[];
  recentLogs: string[];
  nextEventId: number;
};

declare global {
  // eslint-disable-next-line no-var
  var __sportsModelingTrainingState: TrainingState | undefined;
}

const trainingState: TrainingState = globalThis.__sportsModelingTrainingState || {
  running: false,
  events: [],
  recentLogs: [],
  nextEventId: 1,
};
globalThis.__sportsModelingTrainingState = trainingState;

const PROGRESS_PREFIX = "TRAIN_PROGRESS::";
const MAX_EVENTS = 320;
const MAX_LOG_LINES = 80;

function repoRootPath(): string {
  return path.resolve(process.cwd(), "..");
}

function configPathForLeague(league: LeagueCode): string {
  const envOverride = league === "NBA" ? process.env.NBA_CONFIG_PATH : process.env.NHL_CONFIG_PATH;
  const fallback = league === "NBA" ? "configs/nba.yaml" : "configs/nhl.yaml";
  return path.resolve(repoRootPath(), envOverride || fallback);
}

function parseRequestedModels(raw: unknown): ModelName[] | null {
  if (!Array.isArray(raw)) return null;
  const cleaned = raw
    .map((item) => String(item || "").trim().toLowerCase())
    .filter(Boolean)
    .map((item) => item as ModelName);
  if (!cleaned.length) return null;

  const unique: ModelName[] = [];
  for (const token of cleaned) {
    if (!ALL_MODEL_NAMES.includes(token)) {
      throw new Error(`Unknown model '${token}'.`);
    }
    if (!unique.includes(token)) {
      unique.push(token);
    }
  }
  return unique.length ? unique : null;
}

function appendLog(line: string): void {
  trainingState.recentLogs.push(line.slice(-500));
  if (trainingState.recentLogs.length > MAX_LOG_LINES) {
    trainingState.recentLogs.splice(0, trainingState.recentLogs.length - MAX_LOG_LINES);
  }
}

function pushEvent(payload: Partial<ProgressEvent> & { kind?: string; stage?: string; status?: string }): void {
  const event: ProgressEvent = {
    id: trainingState.nextEventId,
    ts_utc: typeof payload.ts_utc === "string" ? payload.ts_utc : new Date().toISOString(),
    kind: typeof payload.kind === "string" ? payload.kind : "pipeline",
    stage: typeof payload.stage === "string" ? payload.stage : "update",
    status: typeof payload.status === "string" ? payload.status : "started",
    message: typeof payload.message === "string" ? payload.message : undefined,
    model: typeof payload.model === "string" ? payload.model : undefined,
    phase: typeof payload.phase === "string" ? payload.phase : undefined,
    fold: typeof payload.fold === "number" ? payload.fold : undefined,
    fold_total: typeof payload.fold_total === "number" ? payload.fold_total : undefined,
    model_total: typeof payload.model_total === "number" ? payload.model_total : undefined,
    selected_models: Array.isArray(payload.selected_models)
      ? payload.selected_models.map((item) => String(item))
      : undefined,
  };

  trainingState.nextEventId += 1;
  trainingState.events.push(event);
  if (trainingState.events.length > MAX_EVENTS) {
    trainingState.events.splice(0, trainingState.events.length - MAX_EVENTS);
  }
}

function parseProgressLine(line: string): Partial<ProgressEvent> | null {
  const markerIdx = line.indexOf(PROGRESS_PREFIX);
  if (markerIdx === -1) return null;
  const raw = line.slice(markerIdx + PROGRESS_PREFIX.length).trim();
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return {
      ts_utc: typeof parsed.ts_utc === "string" ? parsed.ts_utc : new Date().toISOString(),
      kind: typeof parsed.kind === "string" ? parsed.kind : "pipeline",
      stage: typeof parsed.stage === "string" ? parsed.stage : "update",
      status: typeof parsed.status === "string" ? parsed.status : "started",
      message: typeof parsed.message === "string" ? parsed.message : undefined,
      model: typeof parsed.model === "string" ? parsed.model : undefined,
      phase: typeof parsed.phase === "string" ? parsed.phase : undefined,
      fold: typeof parsed.fold === "number" ? parsed.fold : undefined,
      fold_total: typeof parsed.fold_total === "number" ? parsed.fold_total : undefined,
      model_total: typeof parsed.model_total === "number" ? parsed.model_total : undefined,
      selected_models: Array.isArray(parsed.selected_models)
        ? parsed.selected_models.map((item) => String(item))
        : undefined,
    };
  } catch {
    return null;
  }
}

function wireStream(stream: NodeJS.ReadableStream, source: "stdout" | "stderr"): void {
  let buffer = "";
  stream.on("data", (chunk: Buffer | string) => {
    buffer += chunk.toString();

    while (buffer.includes("\n")) {
      const newlineIndex = buffer.indexOf("\n");
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;

      const progress = parseProgressLine(line);
      if (progress) {
        pushEvent(progress);
      } else {
        appendLog(`[${source}] ${line}`);
      }
    }
  });

  stream.on("end", () => {
    const remainder = buffer.trim();
    if (!remainder) return;
    const progress = parseProgressLine(remainder);
    if (progress) {
      pushEvent(progress);
    } else {
      appendLog(`[${source}] ${remainder}`);
    }
  });
}

function summarizeState() {
  const events = [...trainingState.events];
  const modelEvents = trainingState.events.filter((event) => event.kind === "model");
  const latestEvent = events[events.length - 1] || null;
  const latestModelEvent = modelEvents[modelEvents.length - 1] || null;
  const requestedModels = trainingState.requestedModels || [];
  const completedModels = new Set(
    modelEvents
      .filter((event) => event.status === "completed" || event.status === "skipped")
      .map((event) => event.model)
      .filter(Boolean)
  );
  const completedModelCount = requestedModels.length
    ? requestedModels.filter((model) => completedModels.has(model)).length
    : completedModels.size;

  return {
    running: trainingState.running,
    league: trainingState.league,
    requested_models: requestedModels,
    started_at_utc: trainingState.startedAtUtc,
    finished_at_utc: trainingState.finishedAtUtc,
    exit_code: trainingState.exitCode,
    error: trainingState.error,
    latest_event: latestEvent,
    latest_model_event: latestModelEvent,
    model_total: requestedModels.length || undefined,
    completed_models: completedModelCount,
    recent_logs: trainingState.recentLogs,
    events,
    model_events: modelEvents,
  };
}

function startTrainingProcess(league: LeagueCode, configPath: string, models: ModelName[] | null): void {
  trainingState.running = true;
  trainingState.league = league;
  trainingState.requestedModels = models || [...ALL_MODEL_NAMES];
  trainingState.startedAtUtc = new Date().toISOString();
  trainingState.finishedAtUtc = undefined;
  trainingState.exitCode = undefined;
  trainingState.error = undefined;
  trainingState.events = [];
  trainingState.recentLogs = [];
  trainingState.nextEventId = 1;

  pushEvent({
    kind: "pipeline",
    stage: "train_request",
    status: "started",
    message: `Training requested for ${league}: ${(models || ALL_MODEL_NAMES).join(", ")}`,
    selected_models: models || [...ALL_MODEL_NAMES],
  });

  const args = ["-m", "src.cli", "train", "--config", configPath];
  if (models && models.length) {
    args.push("--models", models.join(","));
  }

  const child = spawn("python3", args, {
    cwd: repoRootPath(),
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  wireStream(child.stdout, "stdout");
  wireStream(child.stderr, "stderr");

  child.on("error", (error) => {
    trainingState.running = false;
    trainingState.finishedAtUtc = new Date().toISOString();
    trainingState.exitCode = null;
    trainingState.error = error.message;
    pushEvent({
      kind: "pipeline",
      stage: "train_command",
      status: "failed",
      message: `Training process failed to start: ${error.message}`,
    });
  });

  child.on("close", (code) => {
    trainingState.running = false;
    trainingState.finishedAtUtc = new Date().toISOString();
    trainingState.exitCode = code;

    if (code === 0) {
      pushEvent({
        kind: "pipeline",
        stage: "train_command",
        status: "completed",
        message: "Training process finished successfully",
      });
      return;
    }

    trainingState.error = `Training process exited with code ${code}`;
    pushEvent({
      kind: "pipeline",
      stage: "train_command",
      status: "failed",
      message: trainingState.error,
    });
  });
}

export async function GET(request: Request) {
  const league = leagueFromRequest(request);
  const state = summarizeState();
  return NextResponse.json({ ok: true, requested_league: league, ...state });
}

export async function POST(request: Request) {
  const league = leagueFromRequest(request);

  if (trainingState.running) {
    const state = summarizeState();
    const inProgressMessage = `Training already in progress for ${trainingState.league || "another league"}.`;
    const responseState = {
      ...state,
      error: state.error || inProgressMessage,
    };
    return NextResponse.json(
      {
        ok: false,
        requested_league: league,
        ...responseState,
        message: inProgressMessage,
      },
      { status: 409 }
    );
  }

  let models: ModelName[] | null = null;
  try {
    const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
    models = parseRequestedModels(body.models);
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        requested_league: league,
        error: error instanceof Error ? error.message : "Invalid model request payload.",
      },
      { status: 400 }
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

  startTrainingProcess(league, configPath, models);
  return NextResponse.json({ ok: true, requested_league: league, ...summarizeState() });
}
