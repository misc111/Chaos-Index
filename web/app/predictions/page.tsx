"use client";

import { Suspense, useMemo, useState } from "react";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import { type PredictionsResponse } from "@/lib/types";
import {
  displayPredictionModel,
  formatPredictionAsOf,
  formatPredictionDate,
  formatPredictionProbability,
} from "@/lib/predictions-report";
import TeamWithIcon, { TeamMatchup } from "@/components/TeamWithIcon";
import styles from "./predictions.module.css";

const EMPTY_REPORT: PredictionsResponse = {
  league: "NHL",
  as_of_utc: undefined,
  model_columns: [],
  model_trust_notes: {},
  model_summaries: {},
  model_feature_map_updated_at_utc: undefined,
  rows: [],
};

function probabilityTone(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return styles.probNeutral;
  }
  if (value > 0.9) {
    return styles.probDominant;
  }
  if (value >= 0.65) {
    return styles.probStrong;
  }
  if (value >= 0.55) {
    return styles.probLean;
  }
  if (value > 0.45) {
    return styles.probTossup;
  }
  return styles.probAgainst;
}

function PredictionsPageContent() {
  const [team, setTeam] = useState("");
  const league = useLeague();
  const { data: report, isLoading, error } = useDashboardData<PredictionsResponse>(
    "predictions",
    "/api/predictions",
    league,
    EMPTY_REPORT
  );

  const teams = useMemo(
    () => Array.from(new Set(report.rows.map((row) => row.home_team))).sort(),
    [report.rows]
  );
  const selectedTeam = team && teams.includes(team) ? team : "";

  const filteredRows = useMemo(
    () => report.rows.filter((row) => !selectedTeam || row.home_team === selectedTeam),
    [report.rows, selectedTeam]
  );

  const modelEntries = report.model_columns.map((model) => ({
    key: model,
    label: displayPredictionModel(model),
    summary: report.model_summaries[model],
  }));

  return (
    <div className={styles.page}>
      <section className={styles.toolbar}>
        <div className={styles.filterCluster}>
          <div className={styles.filterField}>
            <label htmlFor="predictions-team-filter" className={styles.filterLabel}>
              Home team
            </label>
            <select
              id="predictions-team-filter"
              className={styles.select}
              value={selectedTeam}
              onChange={(event) => setTeam(event.target.value)}
            >
              <option value="">All home teams</option>
              {teams.map((entry) => (
                <option key={entry} value={entry}>
                  {entry}
                </option>
              ))}
            </select>
          </div>
          {selectedTeam ? (
            <button type="button" className={styles.clearButton} onClick={() => setTeam("")}>
              Clear filter
            </button>
          ) : null}
        </div>
      </section>

      <section className={styles.tableCard}>
        <div className={styles.tableHeader}>
          <div>
            <h3 className={styles.sectionTitle}>Next home games</h3>
            <p className={styles.sectionSubtitle}>
              One row per home team when that team&apos;s next scheduled game is at home.
            </p>
          </div>
          <p className={styles.tableCount}>
            {filteredRows.length} {filteredRows.length === 1 ? "team" : "teams"} shown
          </p>
        </div>

        {error ? (
          <div className={styles.errorState}>{error}</div>
        ) : isLoading ? (
          <div className={styles.loadingState}>Loading the latest predictions report...</div>
        ) : filteredRows.length === 0 ? (
          <div className={styles.emptyState}>No home teams match the current filter.</div>
        ) : (
          <>
            <div className={`${styles.tableScroll} ${styles.tableDesktop}`}>
              <table className={styles.reportTable}>
                <thead>
                  <tr>
                    <th>Home Team</th>
                    <th>Away Team</th>
                    <th>Date</th>
                    {modelEntries.map((model) => (
                      <th key={model.key}>{model.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row) => (
                    <tr key={`${row.game_id}-${row.game_date_utc}`}>
                      <td className={styles.teamCell}>
                        <TeamWithIcon league={league} teamCode={row.home_team} label={row.home_team} />
                      </td>
                      <td className={styles.awayCell}>
                        <TeamWithIcon league={league} teamCode={row.away_team} label={row.away_team} />
                      </td>
                      <td className={styles.dateCell}>{formatPredictionDate(row.game_date_utc)}</td>
                      {modelEntries.map((model) => {
                        const value = row.model_win_probabilities?.[model.key];
                        return (
                          <td key={model.key} className={styles.metricCell}>
                            <span className={`${styles.probPill} ${probabilityTone(value)}`}>
                              {formatPredictionProbability(value)}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className={styles.tableMobile}>
              <div className={styles.mobileCardList}>
                {filteredRows.map((row) => (
                  <article key={`${row.game_id}-${row.game_date_utc}-mobile`} className={styles.mobileCard}>
                    <div className={styles.mobileCardTop}>
                      <div>
                        <p className={styles.mobileCardEyebrow}>Matchup</p>
                        <h4 className={styles.mobileCardTitle}>
                          <TeamMatchup
                            league={league}
                            awayTeamCode={row.away_team}
                            homeTeamCode={row.home_team}
                            awayLabel={row.away_team}
                            homeLabel={row.home_team}
                            size="md"
                          />
                        </h4>
                      </div>
                      <span className={`${styles.probPill} ${probabilityTone(row.ensemble_prob_home_win)}`}>
                        Ensemble {formatPredictionProbability(row.ensemble_prob_home_win)}
                      </span>
                    </div>

                    <div className={styles.mobileMetaGrid}>
                      <div className={styles.mobileMetaItem}>
                        <span className={styles.mobileMetaLabel}>Date</span>
                        <span className={styles.mobileMetaValue}>{formatPredictionDate(row.game_date_utc)}</span>
                      </div>
                      <div className={styles.mobileMetaItem}>
                        <span className={styles.mobileMetaLabel}>Home</span>
                        <span className={styles.mobileMetaValue}>
                          <TeamWithIcon league={league} teamCode={row.home_team} label={row.home_team} />
                        </span>
                      </div>
                      <div className={styles.mobileMetaItem}>
                        <span className={styles.mobileMetaLabel}>Away</span>
                        <span className={styles.mobileMetaValue}>
                          <TeamWithIcon league={league} teamCode={row.away_team} label={row.away_team} />
                        </span>
                      </div>
                    </div>

                    <div className={styles.mobileModelGrid}>
                      {modelEntries.map((model) => {
                        const value = row.model_win_probabilities?.[model.key];
                        return (
                          <div key={model.key} className={styles.mobileModelItem}>
                            <span className={styles.mobileModelLabel}>{model.label}</span>
                            <span className={`${styles.probPill} ${probabilityTone(value)}`}>
                              {formatPredictionProbability(value)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </>
        )}
      </section>

      <section className={styles.notesSection}>
        <div>
          <h3 className={styles.sectionTitle}>Model guide</h3>
          <p className={styles.sectionSubtitle}>
            Live model notes plus the current active inputs from the {report.league} feature map
            {report.model_feature_map_updated_at_utc
              ? `, updated ${formatPredictionAsOf(report.model_feature_map_updated_at_utc)}.`
              : "."}
          </p>
        </div>
        <div className={styles.notesGrid}>
          {modelEntries.map((model) => (
            <article
              key={model.key}
              className={`${styles.noteCard} ${model.summary?.active_feature_count ? styles.noteCardFeatureMap : ""}`}
            >
              <div className={styles.noteHeader}>
                <p className={styles.noteLabel}>{model.label}</p>
                {model.summary?.active_feature_count ? (
                  <span className={styles.noteMetric}>{model.summary.active_feature_count} inputs</span>
                ) : null}
              </div>
              {model.summary?.headline ? <p className={styles.noteHeadline}>{model.summary.headline}</p> : null}
              <p className={styles.noteText}>{model.summary?.trust_note || report.model_trust_notes[model.key]}</p>
              {model.summary?.active_features?.length ? (
                <div className={styles.featureGroup}>
                  <p className={styles.featureLabel}>Current inputs</p>
                  <div className={styles.featureList}>
                    {model.summary.active_features.slice(0, 8).map((feature) => (
                      <code key={`${model.key}-${feature}`} className={styles.featureChip}>
                        {feature}
                      </code>
                    ))}
                    {model.summary.active_features.length > 8 ? (
                      <span className={styles.featureOverflow}>
                        +{model.summary.active_features.length - 8} more
                      </span>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function PredictionsPage() {
  return (
    <Suspense fallback={<p className="small">Loading predictions report...</p>}>
      <PredictionsPageContent />
    </Suspense>
  );
}
