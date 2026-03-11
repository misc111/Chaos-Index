"use client";

import { useMemo, useState } from "react";
import styles from "@/components/ModelBetReplayExplorer.module.css";
import { formatUsd } from "@/lib/currency";
import { DEFAULT_BET_STRATEGY, getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";
import { displayPredictionModel, orderPredictionModels } from "@/lib/predictions-report";
import type { ModelReplayRunRow } from "@/lib/types";

type Props = {
  runs: ModelReplayRunRow[];
  defaultStrategy?: BetStrategy;
  comparisonStrategy?: BetStrategy;
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDateLabel(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) return "—";
  const parsed = new Date(raw.includes("T") ? raw : `${raw}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return raw;
  return raw.includes("T")
    ? parsed.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
    : parsed.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function summarizeFeatureSetToken(value?: string | null): string {
  const token = String(value || "").trim();
  if (!token) return "untracked";
  if (token.length <= 18) return token;
  return `${token.slice(0, 10)}...${token.slice(-4)}`;
}

function valueClassName(value: number): string {
  if (value > 0) return `${styles.metricValue} ${styles.metricPositive}`;
  if (value < 0) return `${styles.metricValue} ${styles.metricNegative}`;
  return styles.metricValue;
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value || {}, null, 2);
}

export default function ModelBetReplayExplorer({
  runs,
  defaultStrategy = DEFAULT_BET_STRATEGY,
  comparisonStrategy = "aggressive",
}: Props) {
  const modelFamilies = useMemo(() => orderPredictionModels((runs || []).map((row) => row.model_name)), [runs]);
  const defaultModel = modelFamilies.includes("ensemble") ? "ensemble" : modelFamilies[0] || "";
  const [selectedModelState, setSelectedModelState] = useState(defaultModel);
  const [selectedRunState, setSelectedRunState] = useState("");

  if (!runs || runs.length === 0) {
    return (
      <div className="card">
        <h3 className="title">Versioned Bet Replay</h3>
        <p className={styles.emptyState}>
          No model-version replay is available yet. This view fills in once a league has settled games with both pregame odds and
          versioned forecast history.
        </p>
      </div>
    );
  }

  const selectedModel = modelFamilies.includes(selectedModelState) ? selectedModelState : defaultModel;
  const selectedModelRuns = runs
    .filter((row) => row.model_name === selectedModel)
    .slice()
    .sort((left, right) => Date.parse(String(right.created_at_utc || right.last_replay_date_central || "")) - Date.parse(String(left.created_at_utc || left.last_replay_date_central || "")));
  const selectedRun =
    selectedModelRuns.find((row) => row.model_run_id === selectedRunState) ||
    selectedModelRuns[0] ||
    runs[0];
  const balancedConfig = getBetStrategyConfig(defaultStrategy);
  const aggressiveConfig = getBetStrategyConfig(comparisonStrategy);
  const selectedModelLabel = displayPredictionModel(selectedModel);

  return (
    <div className="grid">
      <section className={`card ${styles.heroCard}`}>
        <p className={styles.eyebrow}>Versioned Replay</p>
        <h3 className="title" style={{ marginTop: 10 }}>See Which Model Snapshot Would Have Made Which Bets</h3>
        <p className={styles.heroText}>
          Pick a model family, then a dated run. Each card below replays only the games that specific model version actually forecasted
          while it was live, then applies the same fixed betting rules so the differences come from the model, not from moving strategy
          thresholds.
        </p>
        <div className={styles.strategyCallout}>
          <p className={styles.strategyCalloutTitle}>
            {balancedConfig.label} is the default lens, with {aggressiveConfig.label.toLowerCase()} shown beside it.
          </p>
          <p className={styles.strategyCalloutText}>
            The selected run also shows its feature-set token, full feature list, training fingerprint, and game-by-game bet outcomes.
          </p>
        </div>
      </section>

      <section className="card">
        <div className={styles.toggleStack}>
          <span className={styles.toggleLabel}>Model family</span>
          <div className={styles.toggleRow}>
            {modelFamilies.map((model) => {
              const count = runs.filter((row) => row.model_name === model).length;
              return (
                <button
                  key={model}
                  type="button"
                  className={`${styles.toggleButton} ${selectedModel === model ? styles.toggleButtonActive : ""}`}
                  onClick={() => {
                    setSelectedModelState(model);
                    setSelectedRunState("");
                  }}
                >
                  <span className={styles.toggleTitle}>{displayPredictionModel(model)}</span>
                  <span className={styles.toggleNote}>{count} replayable run{count === 1 ? "" : "s"}</span>
                </button>
              );
            })}
          </div>
        </div>
      </section>

      <section className="card">
        <div style={{ display: "grid", gap: 10 }}>
          <h3 className="title" style={{ margin: 0 }}>{selectedModelLabel} Run Timeline</h3>
          <p className={styles.heroText}>
            Earlier and later versions stay separated, so we can answer “what did this model version actually do?” instead of averaging
            across changing feature sets.
          </p>
        </div>

        <div className={styles.runGrid} style={{ marginTop: 18 }}>
          {selectedModelRuns.map((run) => (
            <button
              key={run.model_run_id}
              type="button"
              className={`${styles.runButton} ${selectedRun.model_run_id === run.model_run_id ? styles.runButtonActive : ""}`}
              onClick={() => setSelectedRunState(run.model_run_id)}
            >
              <div className={styles.runHeader}>
                <div>
                  <p className={styles.runDate}>{formatDateLabel(run.created_at_utc || run.last_replay_date_central)}</p>
                  <p className={styles.runMeta}>
                    {run.is_latest_version ? "latest scored version" : `version ${run.version_rank || "?"}`} · {run.replayable_games} replayable games
                  </p>
                </div>
                <span className={styles.badge}>{summarizeFeatureSetToken(run.feature_set_version)}</span>
              </div>

              <div className={styles.runMetrics}>
                <div className={styles.metricRow}>
                  <span className={styles.metricLabel}>{balancedConfig.label} P/L</span>
                  <span className={valueClassName(run.strategies.riskAdjusted.total_profit)}>
                    {formatUsd(run.strategies.riskAdjusted.total_profit, { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div className={styles.metricRow}>
                  <span className={styles.metricLabel}>{aggressiveConfig.label} P/L</span>
                  <span className={valueClassName(run.strategies.aggressive.total_profit)}>
                    {formatUsd(run.strategies.aggressive.total_profit, { minimumFractionDigits: 2 })}
                  </span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </section>

      <div className={styles.detailGrid}>
        <section className={`card ${styles.detailCard}`}>
          <div>
            <h3 className="title" style={{ marginBottom: 8 }}>
              {selectedModelLabel} as of {formatDateLabel(selectedRun.created_at_utc || selectedRun.last_replay_date_central)}
            </h3>
            <p className={styles.heroText}>
              This is the concrete replay for one model snapshot: what it would have bet, how those wagers settled, and whether the
              more aggressive sizing meaningfully changed the path.
            </p>
          </div>

          <div className={styles.badgeRow}>
            <span className={styles.badge}>Run ID: {selectedRun.model_run_id}</span>
            <span className={styles.badge}>Feature set: {summarizeFeatureSetToken(selectedRun.feature_set_version)}</span>
            <span className={styles.badge}>{selectedRun.feature_count} features</span>
            <span className={styles.badge}>{selectedRun.replayable_games} replayable games</span>
            {selectedRun.scored_games ? <span className={styles.badge}>{selectedRun.scored_games} scored games</span> : null}
          </div>

          <div className={styles.summaryGrid}>
            <article className={styles.summaryTile}>
              <p className={styles.summaryLabel}>{balancedConfig.label} P/L</p>
              <p className={`${styles.summaryValue} ${selectedRun.strategies.riskAdjusted.total_profit > 0 ? styles.metricPositive : selectedRun.strategies.riskAdjusted.total_profit < 0 ? styles.metricNegative : ""}`}>
                {formatUsd(selectedRun.strategies.riskAdjusted.total_profit, { minimumFractionDigits: 2 })}
              </p>
              <p className={styles.summarySubtext}>
                {selectedRun.strategies.riskAdjusted.wins}-{selectedRun.strategies.riskAdjusted.losses} record on{" "}
                {selectedRun.strategies.riskAdjusted.suggested_bets} bets
              </p>
            </article>

            <article className={styles.summaryTile}>
              <p className={styles.summaryLabel}>{balancedConfig.label} ROI</p>
              <p className={styles.summaryValue}>{formatPercent(selectedRun.strategies.riskAdjusted.roi)}</p>
              <p className={styles.summarySubtext}>
                Risked {formatUsd(selectedRun.strategies.riskAdjusted.total_risked, { minimumFractionDigits: 2 })}
              </p>
            </article>

            <article className={styles.summaryTile}>
              <p className={styles.summaryLabel}>{aggressiveConfig.label} P/L</p>
              <p className={`${styles.summaryValue} ${selectedRun.strategies.aggressive.total_profit > 0 ? styles.metricPositive : selectedRun.strategies.aggressive.total_profit < 0 ? styles.metricNegative : ""}`}>
                {formatUsd(selectedRun.strategies.aggressive.total_profit, { minimumFractionDigits: 2 })}
              </p>
              <p className={styles.summarySubtext}>
                {selectedRun.strategies.aggressive.wins}-{selectedRun.strategies.aggressive.losses} record on{" "}
                {selectedRun.strategies.aggressive.suggested_bets} bets
              </p>
            </article>

            <article className={styles.summaryTile}>
              <p className={styles.summaryLabel}>{aggressiveConfig.label} ROI</p>
              <p className={styles.summaryValue}>{formatPercent(selectedRun.strategies.aggressive.roi)}</p>
              <p className={styles.summarySubtext}>
                Risked {formatUsd(selectedRun.strategies.aggressive.total_risked, { minimumFractionDigits: 2 })}
              </p>
            </article>
          </div>

          <div className={styles.tableWrap}>
            <table className={styles.betTable}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th>{balancedConfig.label}</th>
                  <th>{aggressiveConfig.label}</th>
                </tr>
              </thead>
              <tbody>
                {selectedRun.bets.map((bet) => (
                  <tr key={`${bet.game_id}-${bet.date_central}`}>
                    <td>{formatDateLabel(bet.date_central)}</td>
                    <td>
                      <div className={styles.matchup}>
                        <strong>{bet.away_team} at {bet.home_team}</strong>
                        <span className={styles.matchupMeta}>
                          Final {bet.away_score ?? "—"}-{bet.home_score ?? "—"} · ML {bet.away_moneyline}/{bet.home_moneyline}
                        </span>
                      </div>
                    </td>
                    <td>
                      <div className={styles.decisionCell}>
                        <strong>{bet.strategies.riskAdjusted.bet_label}</strong>
                        <span className={valueClassName(bet.strategies.riskAdjusted.profit)}>
                          {formatUsd(bet.strategies.riskAdjusted.profit, { minimumFractionDigits: 2 })}
                        </span>
                        <span className={styles.decisionReason}>{bet.strategies.riskAdjusted.reason}</span>
                      </div>
                    </td>
                    <td>
                      <div className={styles.decisionCell}>
                        <strong>{bet.strategies.aggressive.bet_label}</strong>
                        <span className={valueClassName(bet.strategies.aggressive.profit)}>
                          {formatUsd(bet.strategies.aggressive.profit, { minimumFractionDigits: 2 })}
                        </span>
                        <span className={styles.decisionReason}>{bet.strategies.aggressive.reason}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <aside className={styles.stack}>
          <section className={`card ${styles.detailCard}`}>
            <h3 className="title">Run Fingerprint</h3>
            <div className={styles.detailsBlock}>
              <div className={styles.detailRow}>
                <span className={styles.detailLabel}>Trained at</span>
                <span className={styles.detailValue}>{formatDateLabel(selectedRun.created_at_utc)}</span>
              </div>
              <div className={styles.detailRow}>
                <span className={styles.detailLabel}>Scored window</span>
                <span className={styles.detailValue}>
                  {formatDateLabel(selectedRun.first_game_date_utc)} to {formatDateLabel(selectedRun.last_game_date_utc)}
                </span>
              </div>
              <div className={styles.detailRow}>
                <span className={styles.detailLabel}>Replay window</span>
                <span className={styles.detailValue}>
                  {formatDateLabel(selectedRun.first_replay_date_central)} to {formatDateLabel(selectedRun.last_replay_date_central)}
                </span>
              </div>
              {selectedRun.avg_log_loss !== null ? (
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Scored metrics</span>
                  <span className={styles.detailValue}>
                    Log loss {selectedRun.avg_log_loss.toFixed(4)} · Brier {selectedRun.avg_brier?.toFixed(4) || "—"} · Accuracy{" "}
                    {selectedRun.accuracy !== null ? formatPercent(selectedRun.accuracy) : "—"}
                  </span>
                </div>
              ) : null}
              {selectedRun.artifact_path ? (
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Artifact path</span>
                  <span className={styles.detailValue}>{selectedRun.artifact_path}</span>
                </div>
              ) : null}
            </div>
          </section>

          <section className={`card ${styles.detailCard}`}>
            <h3 className="title">Feature Set</h3>
            <p className={styles.heroText}>
              This is the exact feature-set token tied to the selected run. Use it to inspect which inputs changed between versions.
            </p>
            <div className={styles.badgeRow}>
              <span className={styles.badge}>{summarizeFeatureSetToken(selectedRun.feature_set_version)}</span>
              <span className={styles.badge}>{selectedRun.feature_count} features</span>
            </div>

            <details className={styles.detailsToggle}>
              <summary>Full feature list</summary>
              <div className={styles.featureList}>
                {selectedRun.feature_columns.length ? (
                  selectedRun.feature_columns.map((feature) => (
                    <span key={feature} className={styles.featureChip}>{feature}</span>
                  ))
                ) : (
                  <p className={styles.emptyState}>No feature column list is stored for this run.</p>
                )}
              </div>
            </details>

            <details className={styles.detailsToggle}>
              <summary>Run params JSON</summary>
              <pre className={styles.codeBlock}>{prettyJson(selectedRun.params)}</pre>
            </details>

            <details className={styles.detailsToggle}>
              <summary>Run metrics JSON</summary>
              <pre className={styles.codeBlock}>{prettyJson(selectedRun.metrics)}</pre>
            </details>
          </section>
        </aside>
      </div>
    </div>
  );
}
