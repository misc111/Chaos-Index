"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { normalizeLeague, withLeague } from "@/lib/league";

type TrainEvent = {
  id: number;
  ts_utc: string;
  kind: string;
  stage: string;
  status: string;
  message?: string;
  model?: string;
  fold?: number;
  fold_total?: number;
};

const MODEL_BUTTONS: Array<{ key: string; label: string }> = [
  { key: "elo_baseline", label: "Train ELO" },
  { key: "dynamic_rating", label: "Train Dynamic Rating" },
  { key: "glm_logit", label: "Train GLM" },
  { key: "gbdt", label: "Train GBDT" },
  { key: "rf", label: "Train RF" },
  { key: "two_stage", label: "Train Two Stage" },
  { key: "goals_poisson", label: "Train Goals Poisson" },
  { key: "simulation_first", label: "Train Simulation" },
  { key: "bayes_bt_state_space", label: "Train Bayes BT" },
  { key: "bayes_goals", label: "Train Bayes Goals" },
  { key: "nn_mlp", label: "Train NN" },
];

type TrainStatusResponse = {
  ok?: boolean;
  running?: boolean;
  league?: string;
  started_at_utc?: string;
  finished_at_utc?: string;
  exit_code?: number | null;
  error?: string;
  latest_model_event?: TrainEvent | null;
  requested_models?: string[];
  model_total?: number;
  completed_models?: number;
  model_events?: TrainEvent[];
};

const POLL_INTERVAL_MS = 1200;

function stageLabel(value?: string): string {
  if (!value) return "Unknown Stage";
  return value
    .split("_")
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function formatTimestamp(value?: string): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function eventSummary(event: TrainEvent): string {
  const model = event.model ? `${event.model} - ` : "";
  const fold =
    typeof event.fold === "number" && typeof event.fold_total === "number"
      ? ` (fold ${event.fold}/${event.fold_total})`
      : "";
  const message = event.message || `${stageLabel(event.stage)} ${event.status}`;
  return `${model}${message}${fold}`;
}

function modelNameLabel(model: string): string {
  const found = MODEL_BUTTONS.find((item) => item.key === model);
  return found ? found.label.replace(/^Train\s+/i, "") : model;
}

function HomePageContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const [trainStatus, setTrainStatus] = useState<TrainStatusResponse | null>(null);
  const [startingModel, setStartingModel] = useState<string | null>(null);
  const [trainActionError, setTrainActionError] = useState("");

  const fetchTrainStatus = useCallback(async () => {
    try {
      const response = await fetch(withLeague("/api/train-models", league), { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as TrainStatusResponse;
      if (!response.ok) {
        throw new Error(payload.error || `Status request failed (${response.status})`);
      }
      setTrainStatus(payload);
      setTrainActionError("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to fetch training status.";
      setTrainActionError(message);
    }
  }, [league]);

  useEffect(() => {
    void fetchTrainStatus();
  }, [fetchTrainStatus]);

  useEffect(() => {
    if (!trainStatus?.running) return;
    const timer = window.setInterval(() => {
      void fetchTrainStatus();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [fetchTrainStatus, trainStatus?.running]);

  const onTrainModels = async (modelKey: string) => {
    setStartingModel(modelKey);
    setTrainActionError("");

    try {
      const response = await fetch(withLeague("/api/train-models", league), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models: [modelKey] }),
      });
      const payload = (await response.json().catch(() => ({}))) as TrainStatusResponse;
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || `Training request failed (${response.status})`);
      }
      setTrainStatus(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to start training right now.";
      setTrainActionError(message);
    } finally {
      setStartingModel(null);
      void fetchTrainStatus();
    }
  };

  const events = trainStatus?.model_events || [];
  const latestEvent = trainStatus?.latest_model_event || (events.length ? events[events.length - 1] : null);
  const recentEvents = useMemo(() => events.slice(-16).reverse(), [events]);
  const modelTotal = Number(trainStatus?.model_total || 0);
  const completedModels = Number(trainStatus?.completed_models || 0);
  const modelProgressPct = modelTotal > 0 ? Math.min(100, Math.round((completedModels / modelTotal) * 100)) : 0;
  const runningOtherLeague =
    Boolean(trainStatus?.running) && Boolean(trainStatus?.league) && trainStatus?.league !== league;
  const requestedModels = trainStatus?.requested_models || [];

  return (
    <div className="grid two">
      <div className="card">
        <h2 className="title">Pipeline</h2>
        <p className="small">
          Daily workflow: fetch {"->"} features {"->"} train/update {"->"} predict {"->"} ingest results {"->"} score {"->"} performance/validation artifacts.
        </p>
        <p className="small">Use Make targets from repo root to run each stage.</p>
        <div className="train-model-grid">
          {MODEL_BUTTONS.map((model) => {
            const isStartingThis = startingModel === model.key;
            return (
              <button
                key={model.key}
                type="button"
                className="train-btn"
                onClick={() => onTrainModels(model.key)}
                disabled={Boolean(startingModel) || Boolean(trainStatus?.running)}
              >
                {isStartingThis ? "Starting..." : model.label}
              </button>
            );
          })}
        </div>
        <p className="small">Each button runs training for one model in {league}.</p>
      </div>
      <div className="card">
        <h2 className="title">Quick Links</h2>
        <p>
          <Link href={withLeague("/predictions", league)}>Upcoming forecasts</Link>
        </p>
        <p>
          <Link href={withLeague("/actual-vs-expected", league)}>Actual vs Expected</Link>
        </p>
        <p>
          <Link href={withLeague("/leaderboard", league)}>Model leaderboard</Link>
        </p>
        <p>
          <Link href={withLeague("/validation", league)}>Validation suite</Link>
        </p>
      </div>
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <div className="train-monitor-top">
          <h2 className="title" style={{ marginBottom: 0 }}>
            Training Monitor
          </h2>
          <span className="small">
            {trainStatus?.running ? "Live" : "Idle"} {trainStatus?.league ? `(${trainStatus.league})` : ""}
          </span>
        </div>
        {requestedModels.length ? (
          <p className="small train-target-models">Requested models: {requestedModels.map(modelNameLabel).join(", ")}</p>
        ) : null}

        {latestEvent ? (
          <p className="small train-stage-note">
            Current model: <strong>{latestEvent.model || "Unknown model"}</strong> | Stage:{" "}
            <strong>{stageLabel(latestEvent.stage)}</strong> ({latestEvent.status})
          </p>
        ) : trainStatus?.running ? (
          <p className="small train-stage-note">Waiting for first model-stage update...</p>
        ) : (
          <p className="small train-stage-note">No training has started in this session yet.</p>
        )}

        {trainStatus?.running ? (
          <div className="train-progress-block">
            <div className="train-progress-track" role="progressbar" aria-valuenow={modelProgressPct} aria-valuemin={0} aria-valuemax={100}>
              <span className="train-progress-fill" style={{ width: `${modelProgressPct}%` }} />
            </div>
            <p className="small train-progress-text">
              {modelTotal > 0
                ? `Model fit progress: ${completedModels}/${modelTotal} (${modelProgressPct}%)`
                : "Model fit progress: preparing model list..."}
            </p>
          </div>
        ) : null}

        {!trainStatus?.running && trainStatus?.finished_at_utc && !trainStatus?.error ? (
          <p className="small">Last completed at {formatTimestamp(trainStatus.finished_at_utc)}</p>
        ) : null}
        {runningOtherLeague ? (
          <p className="small train-error">Another run is active for {trainStatus?.league}. Switch leagues to follow it live.</p>
        ) : null}
        {(trainActionError || trainStatus?.error) && !trainStatus?.running ? (
          <p className="small train-error">{trainActionError || trainStatus?.error}</p>
        ) : null}

        <div className="train-events-window">
          {recentEvents.length ? (
            <ul className="train-events-list">
              {recentEvents.map((event) => (
                <li key={event.id} className="train-event-item">
                  <span className="train-event-time">{formatTimestamp(event.ts_utc)}</span>
                  <span className={`train-event-status train-event-status-${event.status}`}>{event.status}</span>
                  <span className="train-event-message">{eventSummary(event)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="small">Training events will appear here after you start a run.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={<p className="small">Loading...</p>}>
      <HomePageContent />
    </Suspense>
  );
}
