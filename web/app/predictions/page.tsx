"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { type ForecastRow, type PredictionsResponse } from "@/lib/types";
import { normalizeLeague } from "@/lib/league";
import {
  displayPredictionModel,
  formatPredictionAsOf,
  formatPredictionDate,
  formatPredictionProbability,
} from "@/lib/predictions-report";
import { fetchDashboardJson } from "@/lib/static-staging";
import styles from "./predictions.module.css";

function closestGame(rows: ForecastRow[]): ForecastRow | null {
  if (!rows.length) {
    return null;
  }

  return rows.reduce<ForecastRow>((closest, row) => {
    const currentGap = Math.abs(row.ensemble_prob_home_win - 0.5);
    const nextGap = Math.abs(closest.ensemble_prob_home_win - 0.5);
    return currentGap < nextGap ? row : closest;
  }, rows[0]);
}

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
  const [report, setReport] = useState<PredictionsResponse>({
    league: "NHL",
    as_of_utc: undefined,
    model_columns: [],
    model_trust_notes: {},
    rows: [],
  });
  const [team, setTeam] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    let cancelled = false;

    async function loadReport() {
      setIsLoading(true);
      setError("");

      try {
        const payload = await fetchDashboardJson<Partial<PredictionsResponse>>("predictions", "/api/predictions", league);

        if (!cancelled) {
          setReport({
            league: String(payload.league || league),
            as_of_utc: payload.as_of_utc,
            model_columns: Array.isArray(payload.model_columns) ? payload.model_columns : [],
            model_trust_notes: payload.model_trust_notes || {},
            rows: Array.isArray(payload.rows) ? payload.rows : [],
          });
        }
      } catch (fetchError) {
        if (!cancelled) {
          const message =
            fetchError instanceof Error ? fetchError.message : "Unable to load the predictions report right now.";
          setError(message);
          setReport({
            league,
            as_of_utc: undefined,
            model_columns: [],
            model_trust_notes: {},
            rows: [],
          });
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void loadReport();

    return () => {
      cancelled = true;
    };
  }, [league]);

  useEffect(() => {
    setTeam("");
  }, [league]);

  const teams = useMemo(
    () => Array.from(new Set(report.rows.map((row) => row.home_team))).sort(),
    [report.rows]
  );

  useEffect(() => {
    if (!team) {
      return;
    }
    if (!teams.includes(team)) {
      setTeam("");
    }
  }, [team, teams]);

  const filteredRows = useMemo(
    () => report.rows.filter((row) => !team || row.home_team === team),
    [report.rows, team]
  );

  const strongestEdge = filteredRows[0] || null;
  const closestMatchup = useMemo(() => closestGame(filteredRows), [filteredRows]);

  const windowLabel = useMemo(() => {
    const dates = filteredRows
      .map((row) => {
        const parsed = new Date(row.game_date_utc);
        return Number.isNaN(parsed.getTime()) ? null : parsed;
      })
      .filter((value): value is Date => value instanceof Date)
      .sort((a, b) => a.getTime() - b.getTime());

    if (!dates.length) {
      return "No scheduled games";
    }

    const first = formatPredictionDate(dates[0].toISOString());
    const last = formatPredictionDate(dates[dates.length - 1].toISOString());
    return first === last ? first : `${first} to ${last}`;
  }, [filteredRows]);

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroLead}>
          <p className={styles.eyebrow}>{league} report view</p>
          <h2 className={styles.headline}>Predictions</h2>
          <p className={styles.description}>
            A compact report view for teams whose next scheduled game is at home. Every row belongs to the home side,
            column two is the opponent, and every probability cell is interpreted from that home team&apos;s point of
            view.
          </p>
          <div className={styles.heroHighlights}>
            <span className={styles.heroHighlight}>
              <strong>{report.model_columns.length || 0}</strong> active models
            </span>
            <span className={styles.heroHighlight}>
              <strong>{filteredRows.length}</strong> visible home teams
            </span>
            <span className={styles.heroHighlight}>
              <strong>Unique</strong> home rows
            </span>
          </div>
        </div>

        <div className={styles.summaryGrid}>
          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>As Of</p>
            <p className={`${styles.summaryValue} ${styles.summaryValueCompact}`}>
              {formatPredictionAsOf(report.as_of_utc)}
            </p>
            <p className={styles.summaryHint}>Latest snapshot returned by the local forecast database.</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Window</p>
            <p className={`${styles.summaryValue} ${styles.summaryValueCompact}`}>{windowLabel}</p>
            <p className={styles.summaryHint}>Current visible game dates after filtering.</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Strongest Home Edge</p>
            <p className={styles.summaryValue}>
              {strongestEdge ? `${strongestEdge.home_team} ${formatPredictionProbability(strongestEdge.ensemble_prob_home_win)}` : "No data"}
            </p>
            <p className={styles.summaryHint}>
              {strongestEdge ? `${strongestEdge.away_team} at ${strongestEdge.home_team}` : "No games are available in the current filter."}
            </p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Closest Matchup</p>
            <p className={styles.summaryValue}>
              {closestMatchup ? formatPredictionProbability(closestMatchup.ensemble_prob_home_win) : "No data"}
            </p>
            <p className={styles.summaryHint}>
              {closestMatchup ? `${closestMatchup.away_team} at ${closestMatchup.home_team}` : "No games are available in the current filter."}
            </p>
          </article>
        </div>
      </section>

      <section className={styles.toolbar}>
        <div className={styles.filterCluster}>
          <div className={styles.filterField}>
            <label htmlFor="predictions-team-filter" className={styles.filterLabel}>
              Home team
            </label>
            <select
              id="predictions-team-filter"
              className={styles.select}
              value={team}
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
          {team ? (
            <button type="button" className={styles.clearButton} onClick={() => setTeam("")}>
              Clear filter
            </button>
          ) : null}
        </div>
        <p className={styles.toolbarHint}>
          Each row is the subset of teams whose next game is at home. Teams do not repeat, the away side is column two,
          and every percentage is the home team&apos;s win probability.
        </p>
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
          <div className={styles.tableScroll}>
            <table className={styles.reportTable}>
              <thead>
                <tr>
                  <th>Home Team</th>
                  <th>Away Team</th>
                  <th>Date</th>
                  {report.model_columns.map((model) => (
                    <th key={model}>{displayPredictionModel(model)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => (
                  <tr key={`${row.game_id}-${row.game_date_utc}`}>
                    <td className={styles.teamCell}>{row.home_team}</td>
                    <td className={styles.awayCell}>{row.away_team}</td>
                    <td className={styles.dateCell}>{formatPredictionDate(row.game_date_utc)}</td>
                    {report.model_columns.map((model) => {
                      const value = row.model_win_probabilities?.[model];
                      return (
                        <td key={model} className={styles.metricCell}>
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
        )}
      </section>

      <section className={styles.notesSection}>
        <div>
          <h3 className={styles.sectionTitle}>Model guide</h3>
          <p className={styles.sectionSubtitle}>
            Same explanations as the shareable report, turned into scan-friendly cards for the dashboard.
          </p>
        </div>
        <div className={styles.notesGrid}>
          {report.model_columns.map((model) => (
            <article key={model} className={styles.noteCard}>
              <p className={styles.noteLabel}>{displayPredictionModel(model)}</p>
              <p className={styles.noteText}>{report.model_trust_notes[model]}</p>
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
