"use client";

import { useMemo, useState } from "react";
import EnsembleSnapshotBankrollChart from "@/components/EnsembleSnapshotBankrollChart";
import styles from "@/components/EnsembleSnapshotExplorer.module.css";
import { formatUsd } from "@/lib/currency";
import { getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";
import { displayPredictionModel } from "@/lib/predictions-report";
import type { EnsembleSnapshotRow } from "@/lib/types";

type SnapshotStrategyKey = "riskAdjusted" | "aggressive";

type Props = {
  snapshots: EnsembleSnapshotRow[];
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

function summarizeFingerprint(value?: string | null): string {
  const token = String(value || "").trim();
  if (!token) return "untracked";
  if (token.length <= 14) return token;
  return `${token.slice(0, 8)}...${token.slice(-4)}`;
}

function valueClassName(value: number): string {
  if (value > 0) return `${styles.metricValue} ${styles.metricPositive}`;
  if (value < 0) return `${styles.metricValue} ${styles.metricNegative}`;
  return styles.metricValue;
}

function cellClassName(value: number | null): string {
  if (value === null) return styles.matrixCellEmpty;
  if (value > 0) return `${styles.matrixCell} ${styles.matrixCellPositive}`;
  if (value < 0) return `${styles.matrixCell} ${styles.matrixCellNegative}`;
  return styles.matrixCell;
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value || {}, null, 2);
}

function strategyKeyFromPreference(value?: BetStrategy): SnapshotStrategyKey {
  return value === "aggressive" ? "aggressive" : "riskAdjusted";
}

function cumulativeProfitForDate(snapshot: EnsembleSnapshotRow, strategy: SnapshotStrategyKey, dateCentral: string): number | null {
  let cumulative: number | null = null;
  for (const day of snapshot.daily) {
    if (day.date_central > dateCentral) break;
    cumulative = day.strategies[strategy].cumulative_profit;
  }
  return cumulative;
}

export default function EnsembleSnapshotExplorer({
  snapshots,
  defaultStrategy = "riskAdjusted",
  comparisonStrategy = "aggressive",
}: Props) {
  const primaryStrategy = strategyKeyFromPreference(defaultStrategy);
  const secondaryStrategy = strategyKeyFromPreference(comparisonStrategy) === "riskAdjusted" ? "aggressive" : strategyKeyFromPreference(comparisonStrategy);
  const [selectedSnapshotKey, setSelectedSnapshotKey] = useState("");
  const [matrixStrategy, setMatrixStrategy] = useState<SnapshotStrategyKey>(primaryStrategy);

  const sortedSnapshots = useMemo(
    () =>
      snapshots
        .slice()
        .sort(
          (left, right) =>
            Date.parse(String(right.finalized_at_utc || right.activation_date_central || "")) -
            Date.parse(String(left.finalized_at_utc || left.activation_date_central || ""))
        ),
    [snapshots]
  );
  const matrixSnapshots = useMemo(
    () => snapshots.slice().sort((left, right) => left.activation_date_central.localeCompare(right.activation_date_central)),
    [snapshots]
  );

  if (!snapshots || snapshots.length === 0) {
    return (
      <div className="card">
        <h3 className="title">Frozen Ensemble Snapshots</h3>
        <p className={styles.emptyState}>
          No frozen ensemble snapshot history is available yet. This section fills in once the league has both dated ensemble runs and
          settled games with replayable odds.
        </p>
      </div>
    );
  }

  // Keep the entire explorer on one snapshot identity. The top bankroll chart,
  // the timeline cards, and the detailed provenance tables should all be
  // talking about the same frozen model state instead of drifting apart.
  const activeSelectedSnapshotKey =
    sortedSnapshots.some((snapshot) => snapshot.snapshot_key === selectedSnapshotKey) ? selectedSnapshotKey : sortedSnapshots[0].snapshot_key;
  const selectedSnapshot = sortedSnapshots.find((snapshot) => snapshot.snapshot_key === activeSelectedSnapshotKey) || sortedSnapshots[0];
  const primaryConfig = getBetStrategyConfig(primaryStrategy);
  const secondaryConfig = getBetStrategyConfig(secondaryStrategy);
  const matrixStrategyConfig = getBetStrategyConfig(matrixStrategy);

  // The matrix uses the union of tracked dates so each row answers a simple
  // question: "By this date, what would each frozen snapshot have been worth?"
  const matrixDates = Array.from(new Set(matrixSnapshots.flatMap((snapshot) => snapshot.daily.map((day) => day.date_central)))).sort();

  return (
    <div className="grid">
      <EnsembleSnapshotBankrollChart
        snapshots={matrixSnapshots}
        defaultStrategy={defaultStrategy}
        comparisonStrategy={comparisonStrategy}
        selectedSnapshotKey={activeSelectedSnapshotKey}
        onSelectSnapshotKey={setSelectedSnapshotKey}
      />

      <section className={`card ${styles.heroCard}`}>
        <p className={styles.eyebrow}>Frozen Counterfactuals</p>
        <h3 className="title" style={{ marginTop: 10 }}>What If You Had Stopped Recalibrating On A Given Day?</h3>
        <p className={styles.heroText}>
          Each card below is the last truthful pregame ensemble snapshot that existed on its activation day. Once selected, that
          snapshot is frozen and replayed forward through later odds and results, so we can compare cumulative winnings as if no later
          model changes had happened.
        </p>
        <div className={styles.strategyCallout}>
          <p className={styles.strategyCalloutTitle}>
            {primaryConfig.label} is the default lens, with {secondaryConfig.label.toLowerCase()} shown as the comparison point.
          </p>
          <p className={styles.strategyCalloutText}>
            Snapshot identity is anchored to the saved ensemble artifacts, component-model details, and the latest model-affecting git
            commit that existed before the run finalized.
          </p>
        </div>
      </section>

      <section className="card">
        <div style={{ display: "grid", gap: 10 }}>
          <h3 className="title" style={{ margin: 0 }}>Snapshot Timeline</h3>
          <p className={styles.heroText}>
            Click a snapshot to inspect the exact model state that went live, then compare its frozen bankroll path against the later
            snapshots that replaced it.
          </p>
        </div>

        <div className={styles.snapshotGrid}>
          {sortedSnapshots.map((snapshot) => (
            <button
              key={snapshot.snapshot_key}
              type="button"
              className={`${styles.snapshotButton} ${selectedSnapshot.snapshot_key === snapshot.snapshot_key ? styles.snapshotButtonActive : ""}`}
              onClick={() => setSelectedSnapshotKey(snapshot.snapshot_key)}
            >
              <div className={styles.snapshotHeader}>
                <div>
                  <p className={styles.snapshotDate}>Model as of {formatDateLabel(snapshot.activation_date_central)}</p>
                  <p className={styles.snapshotMeta}>
                    finalized {formatDateLabel(snapshot.finalized_at_utc || snapshot.activation_date_central)} · {snapshot.replayable_games} replayable games
                  </p>
                </div>
                <span className={styles.badge}>{summarizeFeatureSetToken(snapshot.feature_set_version)}</span>
              </div>

              <div className={styles.snapshotMetrics}>
                <div className={styles.metricRow}>
                  <span className={styles.metricLabel}>{primaryConfig.label} P/L</span>
                  <span className={valueClassName(snapshot.strategies.riskAdjusted.total_profit)}>
                    {formatUsd(snapshot.strategies.riskAdjusted.total_profit, { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div className={styles.metricRow}>
                  <span className={styles.metricLabel}>{secondaryConfig.label} P/L</span>
                  <span className={valueClassName(snapshot.strategies.aggressive.total_profit)}>
                    {formatUsd(snapshot.strategies.aggressive.total_profit, { minimumFractionDigits: 2 })}
                  </span>
                </div>
              </div>

              <p className={styles.commitSubject}>{snapshot.model_commit?.subject || "No model commit provenance found"}</p>
            </button>
          ))}
        </div>
      </section>

      <section className="card">
        <div className={styles.matrixHeader}>
          <div>
            <h3 className="title" style={{ margin: 0 }}>Cumulative Winnings Matrix</h3>
            <p className={styles.heroText}>
              Read across each row to compare what every frozen snapshot would have been worth by that date.
            </p>
          </div>
          <div className={styles.toggleRow}>
            <button
              type="button"
              className={`${styles.toggleButton} ${matrixStrategy === "riskAdjusted" ? styles.toggleButtonActive : ""}`}
              onClick={() => setMatrixStrategy("riskAdjusted")}
            >
              {primaryConfig.label}
            </button>
            <button
              type="button"
              className={`${styles.toggleButton} ${matrixStrategy === "aggressive" ? styles.toggleButtonActive : ""}`}
              onClick={() => setMatrixStrategy("aggressive")}
            >
              {secondaryConfig.label}
            </button>
          </div>
        </div>

        <div className={styles.tableWrap}>
          <table className={styles.matrixTable}>
            <thead>
              <tr>
                <th>Date</th>
                {matrixSnapshots.map((snapshot) => (
                  <th key={snapshot.snapshot_key}>
                    <div className={styles.matrixSnapshotHeading}>
                      <strong>{formatDateLabel(snapshot.activation_date_central)}</strong>
                      <span>{summarizeFeatureSetToken(snapshot.feature_set_version)}</span>
                      <span>
                        Final {formatUsd(snapshot.strategies[matrixStrategy].total_profit, { minimumFractionDigits: 2 })}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrixDates.map((dateCentral) => (
                <tr key={dateCentral}>
                  <td>{formatDateLabel(dateCentral)}</td>
                  {matrixSnapshots.map((snapshot) => {
                    const value = cumulativeProfitForDate(snapshot, matrixStrategy, dateCentral);
                    return (
                      <td key={`${snapshot.snapshot_key}-${dateCentral}`} className={cellClassName(value)}>
                        {value === null ? "—" : formatUsd(value, { minimumFractionDigits: 2 })}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={styles.matrixFootnote}>
          The matrix shows {matrixStrategyConfig.label.toLowerCase()} cumulative profit through the last settled game on each date.
        </p>
      </section>

      <section className={`card ${styles.detailCard}`}>
        <div>
          <h3 className="title" style={{ marginBottom: 8 }}>
            {displayPredictionModel(selectedSnapshot.model_name)} as of {formatDateLabel(selectedSnapshot.activation_date_central)}
          </h3>
          <p className={styles.heroText}>
            This drilldown shows the exact frozen snapshot that went live, the git/model provenance tied to it, and how that one state
            would have performed from its activation day forward.
          </p>
        </div>

        <div className={styles.badgeRow}>
          <span className={styles.badge}>Run ID: {selectedSnapshot.model_run_id}</span>
          <span className={styles.badge}>Feature set: {summarizeFeatureSetToken(selectedSnapshot.feature_set_version)}</span>
          <span className={styles.badge}>Fingerprint: {summarizeFingerprint(selectedSnapshot.calibration_fingerprint)}</span>
          <span className={styles.badge}>{selectedSnapshot.feature_count} features</span>
          <span className={styles.badge}>{selectedSnapshot.replayable_games} replayable games</span>
          <span className={styles.badge}>{selectedSnapshot.days_tracked} tracked days</span>
        </div>

        <div className={styles.summaryGrid}>
          <article className={styles.summaryTile}>
            <p className={styles.summaryLabel}>{primaryConfig.label} P/L</p>
            <p className={`${styles.summaryValue} ${selectedSnapshot.strategies.riskAdjusted.total_profit > 0 ? styles.metricPositive : selectedSnapshot.strategies.riskAdjusted.total_profit < 0 ? styles.metricNegative : ""}`}>
              {formatUsd(selectedSnapshot.strategies.riskAdjusted.total_profit, { minimumFractionDigits: 2 })}
            </p>
            <p className={styles.summarySubtext}>
              {selectedSnapshot.strategies.riskAdjusted.wins}-{selectedSnapshot.strategies.riskAdjusted.losses} on{" "}
              {selectedSnapshot.strategies.riskAdjusted.suggested_bets} bets
            </p>
          </article>

          <article className={styles.summaryTile}>
            <p className={styles.summaryLabel}>{primaryConfig.label} ROI</p>
            <p className={styles.summaryValue}>{formatPercent(selectedSnapshot.strategies.riskAdjusted.roi)}</p>
            <p className={styles.summarySubtext}>
              Risked {formatUsd(selectedSnapshot.strategies.riskAdjusted.total_risked, { minimumFractionDigits: 2 })}
            </p>
          </article>

          <article className={styles.summaryTile}>
            <p className={styles.summaryLabel}>{secondaryConfig.label} P/L</p>
            <p className={`${styles.summaryValue} ${selectedSnapshot.strategies.aggressive.total_profit > 0 ? styles.metricPositive : selectedSnapshot.strategies.aggressive.total_profit < 0 ? styles.metricNegative : ""}`}>
              {formatUsd(selectedSnapshot.strategies.aggressive.total_profit, { minimumFractionDigits: 2 })}
            </p>
            <p className={styles.summarySubtext}>
              {selectedSnapshot.strategies.aggressive.wins}-{selectedSnapshot.strategies.aggressive.losses} on{" "}
              {selectedSnapshot.strategies.aggressive.suggested_bets} bets
            </p>
          </article>

          <article className={styles.summaryTile}>
            <p className={styles.summaryLabel}>{secondaryConfig.label} ROI</p>
            <p className={styles.summaryValue}>{formatPercent(selectedSnapshot.strategies.aggressive.roi)}</p>
            <p className={styles.summarySubtext}>
              Risked {formatUsd(selectedSnapshot.strategies.aggressive.total_risked, { minimumFractionDigits: 2 })}
            </p>
          </article>
        </div>

        <div className={styles.provenanceGrid}>
          <article className={styles.provenanceCard}>
            <h4 className={styles.provenanceTitle}>Snapshot Provenance</h4>
            <p className={styles.provenanceLine}>Activated for bets: {formatDateLabel(selectedSnapshot.activation_date_central)}</p>
            <p className={styles.provenanceLine}>Finalized: {formatDateLabel(selectedSnapshot.finalized_at_utc)}</p>
            <p className={styles.provenanceLine}>
              Compared through: {selectedSnapshot.compared_through_date_central ? formatDateLabel(selectedSnapshot.compared_through_date_central) : "No settled games yet"}
            </p>
            <p className={styles.provenanceLine}>
              Model commit: {selectedSnapshot.model_commit ? `${selectedSnapshot.model_commit.short_sha} · ${selectedSnapshot.model_commit.subject}` : "Unavailable"}
            </p>
          </article>

          <article className={styles.provenanceCard}>
            <h4 className={styles.provenanceTitle}>Change Window</h4>
            {selectedSnapshot.commit_window.length ? (
              <ul className={styles.commitList}>
                {selectedSnapshot.commit_window.map((commit) => (
                  <li key={`${commit.sha}-${commit.committed_at_utc}`}>
                    <strong>{commit.short_sha}</strong> {commit.subject}
                  </li>
                ))}
              </ul>
            ) : (
              <p className={styles.heroText}>No model-affecting commits were detected in the window before this snapshot finalized.</p>
            )}
          </article>
        </div>

        <div className={styles.tableWrap}>
          <table className={styles.componentTable}>
            <thead>
              <tr>
                <th>Component model</th>
                <th>Role</th>
                <th>Weight</th>
                <th>Train log loss</th>
                <th>Feature count</th>
              </tr>
            </thead>
            <tbody>
                {selectedSnapshot.component_models.map((component) => (
                  <tr key={component.model_name}>
                    <td>{displayPredictionModel(component.model_name)}</td>
                  <td>
                      {component.included_in_ensemble
                        ? "In ensemble"
                        : component.demoted_from_ensemble
                          ? "Demoted"
                          : component.selected_for_training
                          ? "Trained only"
                          : "Untracked"}
                    </td>
                    <td>{component.weight === null ? "—" : formatPercent(component.weight)}</td>
                    <td>
                      {typeof component.train_metrics?.log_loss === "number"
                        ? Number(component.train_metrics.log_loss).toFixed(4)
                        : "—"}
                    </td>
                    <td>{component.feature_count || "—"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>

        <details className={styles.detailSection} open>
          <summary>Daily bankroll path</summary>
          <div className={styles.tableWrap}>
            <table className={styles.detailTable}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Slate</th>
                  <th>{primaryConfig.label} daily</th>
                  <th>{primaryConfig.label} cumulative</th>
                  <th>{secondaryConfig.label} daily</th>
                  <th>{secondaryConfig.label} cumulative</th>
                </tr>
              </thead>
              <tbody>
                {selectedSnapshot.daily.map((day) => (
                  <tr key={day.date_central}>
                    <td>{formatDateLabel(day.date_central)}</td>
                    <td>
                      {day.slate_games} games · {day.strategies.riskAdjusted.suggested_bets}/{day.strategies.aggressive.suggested_bets} bets
                    </td>
                    <td className={valueClassName(day.strategies.riskAdjusted.total_profit)}>
                      {formatUsd(day.strategies.riskAdjusted.total_profit, { minimumFractionDigits: 2 })}
                    </td>
                    <td className={valueClassName(day.strategies.riskAdjusted.cumulative_profit)}>
                      {formatUsd(day.strategies.riskAdjusted.cumulative_profit, { minimumFractionDigits: 2 })}
                    </td>
                    <td className={valueClassName(day.strategies.aggressive.total_profit)}>
                      {formatUsd(day.strategies.aggressive.total_profit, { minimumFractionDigits: 2 })}
                    </td>
                    <td className={valueClassName(day.strategies.aggressive.cumulative_profit)}>
                      {formatUsd(day.strategies.aggressive.cumulative_profit, { minimumFractionDigits: 2 })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>

        <details className={styles.detailSection}>
          <summary>Game-by-game wagers</summary>
          <div className={styles.tableWrap}>
            <table className={styles.detailTable}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th>{primaryConfig.label}</th>
                  <th>{secondaryConfig.label}</th>
                </tr>
              </thead>
              <tbody>
                {selectedSnapshot.bets.map((bet) => (
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
        </details>

        <details className={styles.detailSection}>
          <summary>Feature lists, model mappings, and training fingerprints</summary>
          <div className={styles.jsonGrid}>
            <article>
              <h4 className={styles.jsonTitle}>Selected models</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.selected_models)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Ensemble component models</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.ensemble_component_columns)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Demoted models</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.demoted_models)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Feature columns</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.feature_columns)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Model feature columns</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.model_feature_columns)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Tuning</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.tuning)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Params</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.params)}</pre>
            </article>
            <article>
              <h4 className={styles.jsonTitle}>Metrics</h4>
              <pre className={styles.codeBlock}>{prettyJson(selectedSnapshot.metrics)}</pre>
            </article>
          </div>
        </details>
      </section>
    </div>
  );
}
