import { spawn } from "node:child_process";
import fs from "node:fs";
import { type LeagueCode } from "@/lib/league";
import { getGlobalState, runBufferedProcess, trimLog, wireLineBufferedStream } from "@/lib/server/task-runner";
import { getModelAliases, getTrainableModels, repoRootPath, resolveConfigPathForLeague } from "@/lib/server/manifests";

type RefreshState = {
  running: boolean;
  league?: LeagueCode;
  startedAtUtc?: string;
};

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

type TrainingState = {
  running: boolean;
  league?: LeagueCode;
  requestedModels?: string[];
  startedAtUtc?: string;
  finishedAtUtc?: string;
  exitCode?: number | null;
  error?: string;
  events: ProgressEvent[];
  recentLogs: string[];
  nextEventId: number;
};

const PROGRESS_PREFIX = "TRAIN_PROGRESS::";
const MAX_EVENTS = 320;
const MAX_LOG_LINES = 80;

function requireConfigPath(league: LeagueCode): string {
  const configPath = resolveConfigPathForLeague(league);
  if (!fs.existsSync(configPath)) {
    throw new Error(`Config file not found: ${configPath}`);
  }
  return configPath;
}

export function getRefreshState(key: "__sportsModelingRefreshState" | "__sportsModelingRefreshOddsState"): RefreshState {
  return getGlobalState<RefreshState>(key, { running: false });
}

export async function runSimpleLeagueTask(command: "refresh-data" | "fetch-odds", league: LeagueCode) {
  const configPath = requireConfigPath(league);
  return runBufferedProcess("python3", ["-m", "src.cli", command, "--config", configPath], {
    cwd: repoRootPath(),
    env: process.env,
  });
}

export function getTrainingState(): TrainingState {
  return getGlobalState<TrainingState>("__sportsModelingTrainingState", {
    running: false,
    events: [],
    recentLogs: [],
    nextEventId: 1,
  });
}

function pushEvent(state: TrainingState, payload: Partial<ProgressEvent> & { kind?: string; stage?: string; status?: string }) {
  const event: ProgressEvent = {
    id: state.nextEventId,
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
    selected_models: Array.isArray(payload.selected_models) ? payload.selected_models.map((item) => String(item)) : undefined,
  };
  state.nextEventId += 1;
  state.events.push(event);
  if (state.events.length > MAX_EVENTS) {
    state.events.splice(0, state.events.length - MAX_EVENTS);
  }
}

function appendTrainingLog(state: TrainingState, line: string) {
  state.recentLogs.push(line.slice(-500));
  if (state.recentLogs.length > MAX_LOG_LINES) {
    state.recentLogs.splice(0, state.recentLogs.length - MAX_LOG_LINES);
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
      selected_models: Array.isArray(parsed.selected_models) ? parsed.selected_models.map((item) => String(item)) : undefined,
    };
  } catch {
    return null;
  }
}

export function parseRequestedModels(raw: unknown): string[] | null {
  if (!Array.isArray(raw)) return null;
  const aliases = getModelAliases();
  const supportedModels = getTrainableModels();
  const cleaned = raw.map((item) => aliases[String(item || "").trim().toLowerCase()] || String(item || "").trim().toLowerCase()).filter(Boolean);
  if (!cleaned.length) return null;

  const unique: string[] = [];
  for (const token of cleaned) {
    if (!supportedModels.includes(token)) {
      throw new Error(`Unknown model '${token}'.`);
    }
    if (!unique.includes(token)) {
      unique.push(token);
    }
  }
  return unique.length ? unique : null;
}

export function summarizeTrainingState() {
  const state = getTrainingState();
  const events = [...state.events];
  const modelEvents = state.events.filter((event) => event.kind === "model");
  const latestEvent = events[events.length - 1] || null;
  const latestModelEvent = modelEvents[modelEvents.length - 1] || null;
  const requestedModels = state.requestedModels || [];
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
    running: state.running,
    league: state.league,
    requested_models: requestedModels,
    started_at_utc: state.startedAtUtc,
    finished_at_utc: state.finishedAtUtc,
    exit_code: state.exitCode,
    error: state.error,
    latest_event: latestEvent,
    latest_model_event: latestModelEvent,
    model_total: requestedModels.length || undefined,
    completed_models: completedModelCount,
    recent_logs: state.recentLogs,
    events,
    model_events: modelEvents,
  };
}

export function startTrainingTask(league: LeagueCode, models: string[] | null): void {
  const state = getTrainingState();
  const supportedModels = getTrainableModels();
  const configPath = requireConfigPath(league);
  state.running = true;
  state.league = league;
  state.requestedModels = models || [...supportedModels];
  state.startedAtUtc = new Date().toISOString();
  state.finishedAtUtc = undefined;
  state.exitCode = undefined;
  state.error = undefined;
  state.events = [];
  state.recentLogs = [];
  state.nextEventId = 1;

  pushEvent(state, {
    kind: "pipeline",
    stage: "train_request",
    status: "started",
    message: `Training requested for ${league}: ${(models || supportedModels).join(", ")}`,
    selected_models: models || [...supportedModels],
  });

  const args = ["-m", "src.cli", "train", "--config", configPath];
  if (models?.length) {
    args.push("--models", models.join(","));
  }

  const child = spawn("python3", args, {
    cwd: repoRootPath(),
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  wireLineBufferedStream(child.stdout, (line) => {
    const progress = parseProgressLine(line);
    if (progress) {
      pushEvent(state, progress);
    } else {
      appendTrainingLog(state, `[stdout] ${line}`);
    }
  });
  wireLineBufferedStream(child.stderr, (line) => {
    const progress = parseProgressLine(line);
    if (progress) {
      pushEvent(state, progress);
    } else {
      appendTrainingLog(state, `[stderr] ${line}`);
    }
  });

  child.on("error", (error) => {
    state.running = false;
    state.finishedAtUtc = new Date().toISOString();
    state.exitCode = null;
    state.error = error.message;
    pushEvent(state, {
      kind: "pipeline",
      stage: "train_command",
      status: "failed",
      message: `Training process failed to start: ${error.message}`,
    });
  });

  child.on("close", (code) => {
    state.running = false;
    state.finishedAtUtc = new Date().toISOString();
    state.exitCode = code;
    if (code === 0) {
      pushEvent(state, {
        kind: "pipeline",
        stage: "train_command",
        status: "completed",
        message: "Training process finished successfully",
      });
      return;
    }
    state.error = `Training process exited with code ${code}`;
    pushEvent(state, {
      kind: "pipeline",
      stage: "train_command",
      status: "failed",
      message: state.error,
    });
  });
}

export { trimLog };
