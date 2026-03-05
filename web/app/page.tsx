"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { normalizeLeague, withLeague } from "@/lib/league";
import styles from "./overview.module.css";

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

const QUICK_LINKS: Array<{ href: string; label: string; description: string }> = [
  {
    href: "/predictions",
    label: "Upcoming forecasts",
    description: "Check the next slate, win probabilities, and model output before tip-off.",
  },
  {
    href: "/actual-vs-expected",
    label: "Actual vs expected",
    description: "Compare realized outcomes with forecasted performance and spot drift quickly.",
  },
  {
    href: "/leaderboard",
    label: "Model leaderboard",
    description: "See which models are earning trust across the current evaluation window.",
  },
  {
    href: "/validation",
    label: "Validation suite",
    description: "Open the deeper QA screens when a fresh run needs calibration checks.",
  },
];

const PIPELINE_STEPS: Array<{ label: string; description: string; keywords: string[] }> = [
  {
    label: "Fetch",
    description: "Pull schedules, injuries, odds, and prior results.",
    keywords: ["fetch", "ingest"],
  },
  {
    label: "Features",
    description: "Rebuild the modeling inputs that feed every training run.",
    keywords: ["feature"],
  },
  {
    label: "Train",
    description: "Fit the selected models or run the whole suite.",
    keywords: ["train", "fit"],
  },
  {
    label: "Forecast",
    description: "Generate new probabilities for the upcoming board.",
    keywords: ["predict", "forecast"],
  },
  {
    label: "Results",
    description: "Ingest outcomes back into the tracking pipeline.",
    keywords: ["result", "settle"],
  },
  {
    label: "Validate",
    description: "Score performance, calibration, and stability artifacts.",
    keywords: ["score", "validation", "performance", "calibration"],
  },
];

type TrainStatusResponse = {
  ok?: boolean;
  running?: boolean;
  league?: string;
  started_at_utc?: string;
  finished_at_utc?: string;
  exit_code?: number | null;
  error?: string;
  latest_event?: TrainEvent | null;
  latest_model_event?: TrainEvent | null;
  requested_models?: string[];
  model_total?: number;
  completed_models?: number;
  events?: TrainEvent[];
  model_events?: TrainEvent[];
};

const POLL_INTERVAL_MS = 1200;

function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

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
  const model = event.model ? `${modelNameLabel(event.model)} - ` : event.kind === "pipeline" ? "Pipeline - " : "";
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

function eventStatusClass(status: string): string {
  switch (status) {
    case "started":
      return styles.eventStatusStarted;
    case "completed":
      return styles.eventStatusCompleted;
    case "skipped":
      return styles.eventStatusSkipped;
    case "failed":
      return styles.eventStatusFailed;
    default:
      return "";
  }
}

function monitorBadgeClass(running: boolean | undefined, runningOtherLeague: boolean): string {
  if (runningOtherLeague) return styles.statusBusy;
  return running ? styles.statusLive : styles.statusIdle;
}

function stageIsActive(keywords: string[], currentStage?: string): boolean {
  if (!currentStage) return false;
  const normalizedStage = currentStage.toLowerCase();
  return keywords.some((keyword) => normalizedStage.includes(keyword));
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
      return payload;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to fetch training status.";
      setTrainActionError(message);
      return null;
    }
  }, [league]);

  useEffect(() => {
    void fetchTrainStatus();
  }, [fetchTrainStatus]);

  useEffect(() => {
    if (!startingModel && !trainStatus?.running) return;

    let timer: number | null = null;
    let cancelled = false;

    const pollStatus = async () => {
      const nextStatus = await fetchTrainStatus();
      if (cancelled || !nextStatus?.running) return;
      timer = window.setTimeout(() => {
        void pollStatus();
      }, POLL_INTERVAL_MS);
    };

    timer = window.setTimeout(() => {
      void pollStatus();
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [fetchTrainStatus, startingModel, trainStatus?.running]);

  const startTrainingRequest = async (marker: string, models?: string[]) => {
    setStartingModel(marker);
    setTrainActionError("");

    try {
      const response = await fetch(withLeague("/api/train-models", league), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(models && models.length ? { models } : {}),
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

  const onTrainModels = async (modelKey: string) => {
    await startTrainingRequest(modelKey, [modelKey]);
  };

  const onTrainAllModels = async () => {
    await startTrainingRequest("__all__");
  };

  const events = trainStatus?.events || trainStatus?.model_events || [];
  const latestEvent = trainStatus?.latest_event || (events.length ? events[events.length - 1] : null);
  const recentEvents = useMemo(() => events.slice(-16).reverse(), [events]);
  const modelTotal = Number(trainStatus?.model_total || 0);
  const completedModels = Number(trainStatus?.completed_models || 0);
  const modelProgressPct = modelTotal > 0 ? Math.min(100, Math.round((completedModels / modelTotal) * 100)) : 0;
  const isTrainingActive = Boolean(startingModel) || Boolean(trainStatus?.running);
  const runningOtherLeague =
    Boolean(trainStatus?.running) && Boolean(trainStatus?.league) && trainStatus?.league !== league;
  const requestedModels = trainStatus?.requested_models || [];
  const activeRequestedModels = isTrainingActive ? requestedModels : [];
  const requestedSet = useMemo(() => new Set(activeRequestedModels), [activeRequestedModels]);
  const isSingleModelRun = Boolean(trainStatus?.running) && requestedModels.length === 1;
  const isStartingSingleModel = Boolean(startingModel) && startingModel !== "__all__";
  const singleModelInFlight = isSingleModelRun || isStartingSingleModel;
  const activeModelKey = isSingleModelRun ? requestedModels[0] : isStartingSingleModel ? startingModel : null;
  const latestStageLabel = latestEvent ? stageLabel(latestEvent.stage) : "No live stage yet";
  const requestedSummary =
    activeRequestedModels.length === 0
      ? isTrainingActive
        ? "Preparing model list"
        : "Full suite or single model"
      : activeRequestedModels.length === 1
        ? modelNameLabel(activeRequestedModels[0])
        : `${activeRequestedModels.length} models selected`;
  const trainingStatusTitle = runningOtherLeague
    ? `Busy in ${trainStatus?.league}`
    : isTrainingActive
      ? "Training live"
      : "Ready to run";
  const trainingStatusDetail = runningOtherLeague
    ? `A ${trainStatus?.league} run is already active. Switch leagues if you want to follow that job instead.`
    : isTrainingActive
      ? activeRequestedModels.length > 1
        ? `${activeRequestedModels.length} models are currently in scope for this run.`
        : activeRequestedModels.length === 1
          ? `${requestedSummary} is the active training target.`
          : "The pipeline is preparing the selected model set."
      : `Choose from ${MODEL_BUTTONS.length} training targets or launch the full suite.`;
  const latestSummary = latestEvent
    ? eventSummary(latestEvent)
    : isTrainingActive
      ? "Waiting for first training update..."
      : "No training has started in this session yet.";
  const lastCompletedLabel = trainStatus?.finished_at_utc
    ? formatTimestamp(trainStatus.finished_at_utc)
    : "No completed run yet";
  const progressSummary = isTrainingActive
    ? modelTotal > 0
      ? `${completedModels}/${modelTotal} models complete`
      : "Preparing model list..."
    : `${MODEL_BUTTONS.length} models available`;
  const liveBadgeLabel = runningOtherLeague ? `Watching ${trainStatus?.league}` : trainStatus?.running ? "Live" : "Idle";

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroLead}>
          <p className={styles.eyebrow}>{league} command center</p>
          <h2 className={styles.headline}>Run the forecasting pipeline from one clean control surface.</h2>
          <p className={styles.description}>
            Kick off training, keep an eye on the live job, and jump straight into the pages that matter after a new run lands.
          </p>

          <div className={styles.heroHighlights}>
            <span className={styles.heroHighlight}>
              <strong>{trainingStatusTitle}</strong>
              <span>{progressSummary}</span>
            </span>
            <span className={styles.heroHighlight}>
              <strong>{requestedSummary}</strong>
              <span>current scope</span>
            </span>
            <span className={styles.heroHighlight}>
              <strong>{latestStageLabel}</strong>
              <span>latest stage</span>
            </span>
          </div>

          <div className={styles.pipelineStrip} aria-label="Pipeline stages">
            {PIPELINE_STEPS.map((step, index) => (
              <div
                key={step.label}
                className={cx(styles.pipelineStage, stageIsActive(step.keywords, latestEvent?.stage) && styles.pipelineStageActive)}
              >
                <span className={styles.pipelineIndex}>{index + 1}</span>
                <div>
                  <p className={styles.pipelineLabel}>{step.label}</p>
                  <p className={styles.pipelineDescription}>{step.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <aside className={styles.summaryGrid}>
          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Training status</p>
            <p className={styles.summaryValue}>{trainingStatusTitle}</p>
            <p className={styles.summaryHint}>{trainingStatusDetail}</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Run scope</p>
            <p className={cx(styles.summaryValue, styles.summaryValueCompact)}>{requestedSummary}</p>
            <p className={styles.summaryHint}>Switch between a full retrain and a single-model pass without leaving overview.</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Latest update</p>
            <p className={cx(styles.summaryValue, styles.summaryValueCompact)}>{latestStageLabel}</p>
            <p className={styles.summaryHint}>{latestSummary}</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Last completion</p>
            <p className={cx(styles.summaryValue, styles.summaryValueCompact)}>{lastCompletedLabel}</p>
            <p className={styles.summaryHint}>Use validation after a fresh run to check calibration and backtest integrity.</p>
          </article>
        </aside>
      </section>

      <section className={styles.workspaceGrid}>
        <div className={styles.actionPanel}>
          <div className={styles.panelHeader}>
            <div>
              <p className={styles.panelEyebrow}>Model actions</p>
              <h3 className={styles.sectionTitle}>Launch training runs</h3>
            </div>
            <span className={styles.panelTag}>{isTrainingActive ? "Run in progress" : "Ready"}</span>
          </div>

          <p className={styles.sectionSubtitle}>
            Use the full suite when you want a complete refresh, or target a single model for faster iteration in {league}.
          </p>

          <div className={styles.primaryActionRow}>
            <button
              type="button"
              className={cx(
                styles.primaryActionButton,
                trainStatus?.running && requestedModels.length > 1 && styles.buttonActive,
                singleModelInFlight && styles.buttonMuted,
              )}
              onClick={onTrainAllModels}
              disabled={isTrainingActive}
            >
              {startingModel === "__all__"
                ? "Starting..."
                : trainStatus?.running && requestedModels.length > 1
                  ? "Training All..."
                  : "Train All Models"}
            </button>
            <p className={styles.primaryActionHint}>
              Recommended for full-dashboard refreshes and post-ingest model QA.
            </p>
          </div>

          <div className={styles.modelGrid}>
            {MODEL_BUTTONS.map((model) => {
              const isStartingThis = startingModel === model.key;
              const isActiveModel = requestedSet.has(model.key) || activeModelKey === model.key;
              const isGreyedOut = singleModelInFlight && !isActiveModel;
              return (
                <button
                  key={model.key}
                  type="button"
                  className={cx(styles.modelButton, isActiveModel && styles.buttonActive, isGreyedOut && styles.buttonMuted)}
                  onClick={() => onTrainModels(model.key)}
                  disabled={isTrainingActive}
                >
                  {isStartingThis ? "Starting..." : isActiveModel && trainStatus?.running ? "Training..." : model.label}
                </button>
              );
            })}
          </div>

          <p className={styles.footerNote}>Keep this page for operations. Use the linked QA pages once a run finishes.</p>
        </div>

        <div className={styles.sideStack}>
          <div className={styles.linkPanel}>
            <div className={styles.panelHeader}>
              <div>
                <p className={styles.panelEyebrow}>Fast navigation</p>
                <h3 className={styles.sectionTitle}>Quick links</h3>
              </div>
            </div>
            <div className={styles.linkList}>
              {QUICK_LINKS.map((item) => (
                <Link href={withLeague(item.href, league)} key={item.href} className={styles.linkCard}>
                  <span className={styles.linkTitle}>{item.label}</span>
                  <span className={styles.linkDescription}>{item.description}</span>
                </Link>
              ))}
            </div>
          </div>

          <div className={styles.notePanel}>
            <p className={styles.panelEyebrow}>Current focus</p>
            <h3 className={styles.sectionTitle}>What deserves attention now</h3>
            <p className={styles.noteLead}>{latestSummary}</p>
            <p className={styles.noteBody}>
              {runningOtherLeague
                ? `Another ${trainStatus?.league} run is active. Stay on ${league} for local controls or switch leagues to monitor that job live.`
                : isTrainingActive
                  ? "Keep this page open while the job runs, then move to Validation or Actual vs Expected once the fit completes."
                  : "No run is active. If the data is fresh, jump to Predictions. If a run just finished, head straight to Validation."}
            </p>
          </div>
        </div>
      </section>

      <section className={styles.monitorCard}>
        <div className={styles.monitorTop}>
          <div>
            <p className={styles.panelEyebrow}>Live run monitor</p>
            <h3 className={styles.sectionTitle}>Training timeline</h3>
          </div>
          <span className={cx(styles.statusBadge, monitorBadgeClass(trainStatus?.running, runningOtherLeague))}>
            {liveBadgeLabel} {trainStatus?.league ? `(${trainStatus.league})` : ""}
          </span>
        </div>

        {requestedModels.length ? (
          <div className={styles.requestedRow}>
            <span className={styles.requestedLabel}>Requested models</span>
            {requestedModels.map((model) => (
              <span key={model} className={styles.requestedChip}>
                {modelNameLabel(model)}
              </span>
            ))}
          </div>
        ) : null}

        {latestEvent ? (
          <p className={styles.stageNote}>
            Latest update: <strong>{eventSummary(latestEvent)}</strong> ({stageLabel(latestEvent.stage)} {latestEvent.status})
          </p>
        ) : trainStatus?.running ? (
          <p className={styles.stageNote}>Waiting for the first training update...</p>
        ) : (
          <p className={styles.stageNote}>Training events will appear here after you start a run.</p>
        )}

        {trainStatus?.running ? (
          <div className={styles.progressBlock}>
            <div className={styles.progressTrack} role="progressbar" aria-valuenow={modelProgressPct} aria-valuemin={0} aria-valuemax={100}>
              <span className={styles.progressFill} style={{ width: `${modelProgressPct}%` }} />
            </div>
            <p className={styles.progressText}>
              {modelTotal > 0
                ? `Model fit progress: ${completedModels}/${modelTotal} (${modelProgressPct}%)`
                : "Model fit progress: preparing model list..."}
            </p>
          </div>
        ) : !trainStatus?.error && trainStatus?.finished_at_utc ? (
          <p className={styles.progressText}>Last completed at {formatTimestamp(trainStatus.finished_at_utc)}</p>
        ) : null}

        {runningOtherLeague ? (
          <p className={styles.errorText}>Another run is active for {trainStatus?.league}. Switch leagues to follow it live.</p>
        ) : null}
        {(trainActionError || trainStatus?.error) && !trainStatus?.running ? (
          <p className={styles.errorText}>{trainActionError || trainStatus?.error}</p>
        ) : null}

        <div className={styles.eventsWindow}>
          {recentEvents.length ? (
            <ul className={styles.eventList}>
              {recentEvents.map((event) => (
                <li key={event.id} className={styles.eventItem}>
                  <span className={styles.eventTime}>{formatTimestamp(event.ts_utc)}</span>
                  <span className={cx(styles.eventStatus, eventStatusClass(event.status))}>{event.status}</span>
                  <span className={styles.eventMessage}>{eventSummary(event)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className={styles.emptyState}>No events yet. Start a run to populate the live timeline.</p>
          )}
        </div>
      </section>
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
