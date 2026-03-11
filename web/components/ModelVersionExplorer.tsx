"use client";

import { useState } from "react";
import Charts from "@/components/Charts";
import ModelTable from "@/components/ModelTable";
import { displayPredictionModel, orderPredictionModels } from "@/lib/predictions-report";
import type { ModelRunSummaryRow, TableRow } from "@/lib/types";

type MetricKey = "avg_log_loss" | "avg_brier" | "accuracy";

const METRIC_OPTIONS: { key: MetricKey; label: string; note: string; lowerIsBetter: boolean }[] = [
  { key: "avg_log_loss", label: "Log loss", note: "Lower is better", lowerIsBetter: true },
  { key: "avg_brier", label: "Brier", note: "Lower is better", lowerIsBetter: true },
  { key: "accuracy", label: "Accuracy", note: "Higher is better", lowerIsBetter: false },
];

type FeatureSetSummary = {
  feature_set_version: string;
  scored_runs: number;
  n_games: number;
  avg_log_loss: number;
  avg_brier: number;
  accuracy: number;
  first_created_at_utc: string | null;
  last_created_at_utc: string | null;
};

function sortTimestamp(row: Pick<ModelRunSummaryRow, "created_at_utc" | "last_game_date_utc">): number {
  const fallback = row.last_game_date_utc ? `${row.last_game_date_utc}T23:59:59Z` : "";
  const value = Date.parse(String(row.created_at_utc || fallback || ""));
  return Number.isFinite(value) ? value : 0;
}

function metricValue(row: ModelRunSummaryRow, metric: MetricKey): number {
  if (metric === "accuracy") return Number(row.accuracy || 0);
  if (metric === "avg_brier") return Number(row.avg_brier || 0);
  return Number(row.avg_log_loss || 0);
}

function formatMetricValue(metric: MetricKey, value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (metric === "accuracy") return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(4);
}

function formatDelta(metric: MetricKey, value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (metric === "accuracy") return `${(value * 100).toFixed(1)} pts`;
  return value.toFixed(4);
}

function formatDateLabel(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) return "—";
  const parsed = new Date(raw.includes("T") ? raw : `${raw}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return raw;
  return raw.includes("T")
    ? parsed.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
    : parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function summarizeFeatureSetToken(value?: string | null): string {
  const token = String(value || "").trim();
  if (!token) return "untracked";
  if (token.length <= 18) return token;
  return `${token.slice(0, 10)}...${token.slice(-4)}`;
}

function scoredWindowStart(row: Pick<ModelRunSummaryRow, "first_game_date_central" | "first_game_date_utc">): string | null {
  return row.first_game_date_central || row.first_game_date_utc || null;
}

function scoredWindowEnd(row: Pick<ModelRunSummaryRow, "last_game_date_central" | "last_game_date_utc">): string | null {
  return row.last_game_date_central || row.last_game_date_utc || null;
}

function summarizeFeatureSets(rows: ModelRunSummaryRow[]): FeatureSetSummary[] {
  const groups = new Map<string, FeatureSetSummary>();

  for (const row of rows) {
    const key = String(row.feature_set_version || "").trim() || "untracked";
    const current =
      groups.get(key) ||
      {
        feature_set_version: key,
        scored_runs: 0,
        n_games: 0,
        avg_log_loss: 0,
        avg_brier: 0,
        accuracy: 0,
        first_created_at_utc: null,
        last_created_at_utc: null,
      };

    const weight = Number(row.n_games || 0);
    current.scored_runs += 1;
    current.n_games += weight;
    current.avg_log_loss += Number(row.avg_log_loss || 0) * weight;
    current.avg_brier += Number(row.avg_brier || 0) * weight;
    current.accuracy += Number(row.accuracy || 0) * weight;

    if (!current.first_created_at_utc || sortTimestamp({ created_at_utc: row.created_at_utc, last_game_date_utc: row.last_game_date_utc }) < sortTimestamp({
      created_at_utc: current.first_created_at_utc,
      last_game_date_utc: null,
    })) {
      current.first_created_at_utc = row.created_at_utc || current.first_created_at_utc;
    }
    if (!current.last_created_at_utc || sortTimestamp({ created_at_utc: row.created_at_utc, last_game_date_utc: row.last_game_date_utc }) > sortTimestamp({
      created_at_utc: current.last_created_at_utc,
      last_game_date_utc: null,
    })) {
      current.last_created_at_utc = row.created_at_utc || current.last_created_at_utc;
    }

    groups.set(key, current);
  }

  return Array.from(groups.values())
    .map((row) => ({
      ...row,
      avg_log_loss: row.n_games > 0 ? row.avg_log_loss / row.n_games : 0,
      avg_brier: row.n_games > 0 ? row.avg_brier / row.n_games : 0,
      accuracy: row.n_games > 0 ? row.accuracy / row.n_games : 0,
    }))
    .sort((left, right) => sortTimestamp({ created_at_utc: right.last_created_at_utc, last_game_date_utc: null }) - sortTimestamp({
      created_at_utc: left.last_created_at_utc,
      last_game_date_utc: null,
    }));
}

function describeComparison(latestRun: ModelRunSummaryRow, bestOlderRun: ModelRunSummaryRow | null, metric: MetricKey): string {
  if (!bestOlderRun) {
    return "Only one scored run is available for this model family so far, so there is no older version to compare yet.";
  }

  const latestValue = metricValue(latestRun, metric);
  const olderValue = metricValue(bestOlderRun, metric);
  const metricConfig = METRIC_OPTIONS.find((option) => option.key === metric) || METRIC_OPTIONS[0];
  const latestIsBetter = metricConfig.lowerIsBetter ? latestValue <= olderValue : latestValue >= olderValue;
  const delta = metricConfig.lowerIsBetter ? Math.abs(latestValue - olderValue) : latestValue - olderValue;

  return latestIsBetter
    ? `The latest scored run is ${formatDelta(metric, Math.abs(delta))} better than the best older scored run on ${metricConfig.label.toLowerCase()}.`
    : `The latest scored run is ${formatDelta(metric, Math.abs(delta))} worse than the best older scored run on ${metricConfig.label.toLowerCase()}.`;
}

export default function ModelVersionExplorer({ rows }: { rows: ModelRunSummaryRow[] }) {
  const models = orderPredictionModels((rows || []).map((row) => row.model_name));
  const defaultModel = models.includes("ensemble") ? "ensemble" : models[0] || "";
  const [selectedModelState, setSelectedModelState] = useState(defaultModel);
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>("avg_log_loss");

  if (!rows || rows.length === 0) {
    return (
      <div className="card">
        <h3 className="title">Older Model Versions</h3>
        <p className="small">No scored model-run history is available yet.</p>
      </div>
    );
  }

  const selectedModel = models.includes(selectedModelState) ? selectedModelState : defaultModel;

  const selectedRuns = rows
    .filter((row) => row.model_name === selectedModel)
    .slice()
    .sort((left, right) => sortTimestamp(left) - sortTimestamp(right));

  if (selectedRuns.length === 0) {
    return (
      <div className="card">
        <h3 className="title">Older Model Versions</h3>
        <p className="small">No scored run history is available for this model family yet.</p>
      </div>
    );
  }

  const newestRun = selectedRuns[selectedRuns.length - 1];
  const olderRuns = selectedRuns.slice(0, -1);
  const metricConfig = METRIC_OPTIONS.find((option) => option.key === selectedMetric) || METRIC_OPTIONS[0];
  const bestOlderRun =
    olderRuns.length > 0
      ? olderRuns
          .slice()
          .sort((left, right) => {
            const delta = metricValue(left, selectedMetric) - metricValue(right, selectedMetric);
            return metricConfig.lowerIsBetter ? delta : -delta;
          })[0]
      : null;
  const featureSetSummaries = summarizeFeatureSets(selectedRuns);
  const selectedModelLabel = displayPredictionModel(selectedModel);

  const runTableRows: TableRow[] = selectedRuns
    .slice()
    .sort((left, right) => sortTimestamp(right) - sortTimestamp(left))
    .map((row) => ({
      status: row.is_latest_version ? "latest" : `older v${row.version_rank}`,
      trained_at: formatDateLabel(row.created_at_utc || row.last_game_date_utc),
      feature_set: summarizeFeatureSetToken(row.feature_set_version),
      games_scored: String(row.n_games),
      // Prefer the replay-aligned display date when present. The raw UTC score
      // window can be a day late for late-night games.
      score_window: `${formatDateLabel(scoredWindowStart(row))} to ${formatDateLabel(scoredWindowEnd(row))}`,
      log_loss: formatMetricValue("avg_log_loss", Number(row.avg_log_loss)),
      brier: formatMetricValue("avg_brier", Number(row.avg_brier)),
      accuracy: formatMetricValue("accuracy", Number(row.accuracy)),
    }));

  const featureTableRows: TableRow[] = featureSetSummaries.map((row) => ({
    feature_set: summarizeFeatureSetToken(row.feature_set_version),
    scored_runs: String(row.scored_runs),
    games_scored: String(row.n_games),
    log_loss: formatMetricValue("avg_log_loss", row.avg_log_loss),
    brier: formatMetricValue("avg_brier", row.avg_brier),
    accuracy: formatMetricValue("accuracy", row.accuracy),
    trained_span: `${formatDateLabel(row.first_created_at_utc)} to ${formatDateLabel(row.last_created_at_utc)}`,
  }));

  const comparisonCopy = describeComparison(newestRun, bestOlderRun, selectedMetric);

  return (
    <div className="grid">
      <div className="card">
        <span className="badge">Version replay</span>
        <h3 className="title" style={{ marginTop: 12 }}>See How Older Runs Actually Scored</h3>
        <p className="small">
          Each scored run below is one trained model snapshot, evaluated only on the settled games it actually forecasted after it existed.
          This is the cleanest way to see whether a feature or parameter change coincided with a drop.
        </p>

        <div className="strategy-toggle-stack" style={{ marginTop: 18 }}>
          <span className="strategy-toggle-label">Model family</span>
          <div className="strategy-toggle-row">
            {models.map((model) => {
              const count = rows.filter((row) => row.model_name === model).length;
              return (
                <button
                  key={model}
                  type="button"
                  className={`strategy-toggle-btn ${selectedModel === model ? "active" : ""}`}
                  onClick={() => setSelectedModelState(model)}
                >
                  <span className="strategy-toggle-title">{displayPredictionModel(model)}</span>
                  <span className="strategy-toggle-note">{count} scored runs</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="strategy-toggle-stack" style={{ marginTop: 14 }}>
          <span className="strategy-toggle-label">Compare by</span>
          <div className="strategy-toggle-row">
            {METRIC_OPTIONS.map((metric) => (
              <button
                key={metric.key}
                type="button"
                className={`strategy-toggle-btn ${selectedMetric === metric.key ? "active" : ""}`}
                onClick={() => setSelectedMetric(metric.key)}
              >
                <span className="strategy-toggle-title">{metric.label}</span>
                <span className="strategy-toggle-note">{metric.note}</span>
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 18 }}>
          <span className="badge">Latest: {formatMetricValue(selectedMetric, metricValue(newestRun, selectedMetric))}</span>
          {bestOlderRun ? <span className="badge">Best older: {formatMetricValue(selectedMetric, metricValue(bestOlderRun, selectedMetric))}</span> : null}
          <span className="badge">{selectedRuns.length} scored runs</span>
          <span className="badge">{featureSetSummaries.length} feature-set eras</span>
          <span className="badge">Current feature set: {summarizeFeatureSetToken(newestRun.feature_set_version)}</span>
        </div>

        <p className="small" style={{ marginTop: 14 }}>
          {selectedModelLabel} has scored runs from {formatDateLabel(selectedRuns[0]?.created_at_utc || selectedRuns[0]?.last_game_date_utc)} through{" "}
          {formatDateLabel(newestRun.created_at_utc || newestRun.last_game_date_utc)}. {comparisonCopy}
        </p>
      </div>

      <div className="grid two">
        <Charts
          title={`${selectedModelLabel} ${metricConfig.label.toLowerCase()} by run date`}
          points={selectedRuns.map((row) => ({
            x: row.created_at_utc || row.last_game_date_utc || row.model_run_id,
            y: metricValue(row, selectedMetric),
          }))}
          color="var(--accent)"
        />
        <ModelTable title={`Feature-Set Eras For ${selectedModelLabel}`} rows={featureTableRows} />
      </div>

      <ModelTable title={`Scored Runs For ${selectedModelLabel}`} rows={runTableRows} />
    </div>
  );
}
