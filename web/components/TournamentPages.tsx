"use client";

import { Suspense, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { normalizeLeague } from "@/lib/league";
import type { NestedTournamentResponse, PerformanceResponse, PredictionsResponse, TableRow } from "@/lib/types";
import TeamWithIcon, { TeamMatchup } from "@/components/TeamWithIcon";

const EMPTY_NESTED: NestedTournamentResponse = {
  league: "MLB",
  current_best: null,
  summary: null,
  target_coverage: [],
  family_champions: [],
  inter_family_leaderboard: [],
  feature_coverage_summary: [],
  artifacts: {},
};

const EMPTY_PREDICTIONS: PredictionsResponse = {
  league: "MLB",
  model_columns: [],
  model_trust_notes: {},
  model_summaries: {},
  rows: [],
};

const EMPTY_PERFORMANCE: PerformanceResponse = {
  league: "MLB",
  scores: [],
  run_summaries: [],
  change_points: [],
  replay_runs: [],
  ensemble_snapshots: [],
};

function formatPercent(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return `${(numeric * 100).toFixed(1)}%`;
}

function formatNumber(value: unknown, digits = 4): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return numeric.toFixed(digits);
}

function titleCase(value: unknown): string {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase()) || "—";
}

function getString(row: TableRow | undefined | null, key: string): string {
  const value = row?.[key];
  return value == null ? "" : String(value);
}

function getRowsForTarget(rows: TableRow[], target: string): TableRow[] {
  return rows.filter((row) => String(row.target_name || "") === target);
}

function getTournamentTargets(data: NestedTournamentResponse): string[] {
  const names = new Set<string>();
  for (const row of data.target_coverage || []) {
    if (row.target_name) names.add(String(row.target_name));
  }
  for (const row of data.family_champions || []) {
    if (row.target_name) names.add(String(row.target_name));
  }
  for (const row of data.inter_family_leaderboard || []) {
    if (row.target_name) names.add(String(row.target_name));
  }
  return Array.from(names);
}

function useNestedTournamentData() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  return {
    league,
    ...useDashboardData<NestedTournamentResponse>("nestedTournament", "/api/nested-tournament", league, EMPTY_NESTED),
  };
}

function TournamentFrame({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <div className="notebook-page">
      <section className="notebook-hero compact">
        <div>
          <p className="notebook-kicker">Probability notebook</p>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
      </section>
      {children}
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="mini-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TargetPicker({
  targets,
  activeTarget,
  onSelect,
}: {
  targets: string[];
  activeTarget: string;
  onSelect: (target: string) => void;
}) {
  if (targets.length < 2) return null;
  return (
    <div className="target-switcher" role="tablist" aria-label="Tournament target">
      {targets.map((target) => (
        <button
          type="button"
          role="tab"
          key={target}
          className={target === activeTarget ? "active" : ""}
          onClick={() => onSelect(target)}
          aria-selected={target === activeTarget}
        >
          {titleCase(target)}
        </button>
      ))}
    </div>
  );
}

function TournamentLoading({ title }: { title: string }) {
  return (
    <TournamentFrame title={title} subtitle="Loading the latest MLB model evidence.">
      <section className="notebook-panel">
        <p className="small">Loading...</p>
      </section>
    </TournamentFrame>
  );
}

function IntraFamilyContent() {
  const { data, error, isLoading } = useNestedTournamentData();
  const targets = useMemo(() => getTournamentTargets(data), [data]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const target = targets.includes(selectedTarget) ? selectedTarget : targets[0] || "";
  const coverage = getRowsForTarget(data.target_coverage || [], target)[0];
  const champions = getRowsForTarget(data.family_champions || [], target);

  return (
    <TournamentFrame
      title="Intra-Family Tournament"
      subtitle="Each model family plays its own bracket first. The winner is the cleanest version of that family, not necessarily the final champion."
    >
      {isLoading ? <section className="notebook-panel"><p className="small">Loading brackets...</p></section> : null}
      {error ? <section className="notebook-panel"><p className="small">Failed to load: {error}</p></section> : null}
      {!isLoading && !error ? (
        <>
          <TargetPicker targets={targets} activeTarget={target} onSelect={setSelectedTarget} />

          <section className="notebook-metric-strip">
            <MiniMetric label="Target" value={titleCase(target)} />
            <MiniMetric label="Usable games" value={formatNumber(coverage?.usable_rows, 0)} />
            <MiniMetric label="Families shown" value={String(champions.length)} />
            <MiniMetric label="Run" value={getString(data.summary, "run_id").slice(0, 22) || "—"} />
          </section>

          <section className="notebook-panel">
            <div className="notebook-panel-header">
              <div>
                <p className="notebook-kicker">Family winners</p>
                <h2>Best version inside each family</h2>
              </div>
            </div>
            <div className="notebook-card-grid">
              {champions.length ? (
                champions.map((row) => (
                  <article className="model-bubble" key={`${target}-${getString(row, "candidate_key") || getString(row, "model_name")}`}>
                    <div>
                      <p>{titleCase(row.model_name)}</p>
                      <h3>{titleCase(row.variant_display_name || row.variant_key)}</h3>
                    </div>
                    <div className="model-bubble-stats">
                      <MiniMetric label="Log loss" value={formatNumber(row.log_loss)} />
                      <MiniMetric label="Brier" value={formatNumber(row.brier)} />
                      <MiniMetric label="AUC" value={formatNumber(row.auc ?? row.roc_auc)} />
                    </div>
                    <span className={`status-pill ${row.gate_passed ? "good" : "warn"}`}>
                      {row.gate_passed ? "cleared" : titleCase(row.champion_status || "review")}
                    </span>
                  </article>
                ))
              ) : (
                <p className="small">No family champion rows have been published yet.</p>
              )}
            </div>
          </section>
        </>
      ) : null}
    </TournamentFrame>
  );
}

function InterFamilyContent() {
  const { data, error, isLoading } = useNestedTournamentData();
  const targets = useMemo(() => getTournamentTargets(data), [data]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const target = targets.includes(selectedTarget) ? selectedTarget : targets[0] || "";
  const targetRows = getRowsForTarget(data.inter_family_leaderboard || [], target);
  const fallbackRows = getRowsForTarget(data.family_champions || [], target);
  const visibleRows = targetRows.length ? targetRows : fallbackRows;
  const leader = visibleRows[0];
  const evidence = data.current_best?.evidence_status as TableRow | undefined;

  return (
    <TournamentFrame
      title="Inter-Family Tournament"
      subtitle="After each family has a best candidate, the families meet on the final board. This page keeps that comparison separate from the family brackets."
    >
      {isLoading ? <section className="notebook-panel"><p className="small">Loading final board...</p></section> : null}
      {error ? <section className="notebook-panel"><p className="small">Failed to load: {error}</p></section> : null}
      {!isLoading && !error ? (
        <>
          <TargetPicker targets={targets} activeTarget={target} onSelect={setSelectedTarget} />

          <section className="notebook-metric-strip">
            <MiniMetric label="Target" value={titleCase(target)} />
            <MiniMetric label="Current leader" value={titleCase(leader?.model_name || leader?.candidate_key)} />
            <MiniMetric label="Log loss" value={formatNumber(leader?.log_loss)} />
            <MiniMetric label="Evidence" value={titleCase(evidence?.evidence_stage || data.current_best?.source_status)} />
            <MiniMetric label="Promotion" value={data.current_best?.promotion_eligible ? "Eligible" : "Research only"} />
          </section>

          <section className="notebook-panel">
            <div className="notebook-panel-header">
              <div>
                <p className="notebook-kicker">Final comparison</p>
                <h2>{targetRows.length ? "Inter-family leaderboard" : "Family finalists waiting for final holdout rows"}</h2>
              </div>
            </div>
            <div className="friendly-table-wrap">
              <table className="friendly-table">
                <thead>
                  <tr>
                    <th>Family</th>
                    <th>Variant</th>
                    <th>Status</th>
                    <th>Log loss</th>
                    <th>Brier</th>
                    <th>AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map((row, index) => (
                    <tr key={`${target}-${getString(row, "candidate_key") || `${getString(row, "model_name")}-${index}`}`}>
                      <td>{titleCase(row.model_name)}</td>
                      <td>{titleCase(row.variant_display_name || row.variant_key || row.candidate_key)}</td>
                      <td>
                        <span className={`status-pill ${row.gate_passed ? "good" : "warn"}`}>
                          {row.gate_passed ? "cleared" : titleCase(row.champion_status || "review")}
                        </span>
                      </td>
                      <td>{formatNumber(row.log_loss)}</td>
                      <td>{formatNumber(row.brier)}</td>
                      <td>{formatNumber(row.auc ?? row.roc_auc)}</td>
                    </tr>
                  ))}
                  {!visibleRows.length ? (
                    <tr>
                      <td colSpan={6}>No inter-family rows have been published yet.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </TournamentFrame>
  );
}

function EnsembleSummaryContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const predictions = useDashboardData<PredictionsResponse>("predictions", "/api/predictions", league, EMPTY_PREDICTIONS);
  const performance = useDashboardData<PerformanceResponse>("performance", "/api/performance", league, EMPTY_PERFORMANCE);
  const snapshot = performance.data.ensemble_snapshots?.[0];
  const components = snapshot?.component_models || [];

  return (
    <div className="notebook-page">
      <section className="notebook-hero compact">
        <div>
          <p className="notebook-kicker">Ensemble Summary Table</p>
          <h1>The little model roster</h1>
          <p>One compact table for the ensemble’s current game probabilities, plus the component model lineup.</p>
        </div>
      </section>

      <section className="notebook-metric-strip">
        <MiniMetric label="Games" value={String(predictions.data.rows.length)} />
        <MiniMetric label="Models" value={String(predictions.data.model_columns.length || components.length)} />
        <MiniMetric label="Feature set" value={predictions.data.model_feature_set_version || snapshot?.feature_set_version || "—"} />
        <MiniMetric label="As of" value={String(predictions.data.as_of_utc || snapshot?.finalized_at_utc || "—").slice(0, 16)} />
      </section>

      <section className="notebook-panel">
        <div className="notebook-panel-header">
          <div>
            <p className="notebook-kicker">Probabilities</p>
            <h2>Today’s ensemble board</h2>
          </div>
        </div>
        {predictions.isLoading ? <p className="small">Loading probabilities...</p> : null}
        {predictions.error ? <p className="small">Failed to load: {predictions.error}</p> : null}
        <div className="friendly-table-wrap">
          <table className="friendly-table">
            <thead>
              <tr>
                <th>Matchup</th>
                <th>Pick</th>
                <th>Home win</th>
                <th>Market</th>
                <th>Models</th>
              </tr>
            </thead>
            <tbody>
              {predictions.data.rows.map((row) => (
                <tr key={row.game_id}>
                  <td>
                    <TeamMatchup
                      league={league}
                      awayTeamCode={row.away_team}
                      homeTeamCode={row.home_team}
                      awayLabel={row.away_team}
                      homeLabel={row.home_team}
                      size="md"
                    />
                  </td>
                  <td>
                    <TeamWithIcon league={league} teamCode={row.predicted_winner} label={row.predicted_winner} />
                  </td>
                  <td>{formatPercent(row.ensemble_prob_home_win)}</td>
                  <td>{row.moneyline_book || "—"}</td>
                  <td>{Object.keys(row.model_win_probabilities || {}).length}</td>
                </tr>
              ))}
              {!predictions.data.rows.length && !predictions.isLoading ? (
                <tr>
                  <td colSpan={5}>No prediction rows have been published yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="notebook-panel">
        <div className="notebook-panel-header">
          <div>
            <p className="notebook-kicker">Component lineup</p>
            <h2>Who is in the ensemble?</h2>
          </div>
        </div>
        <div className="friendly-table-wrap">
          <table className="friendly-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Included</th>
                <th>Weight</th>
                <th>Features</th>
              </tr>
            </thead>
            <tbody>
              {components.map((row) => (
                <tr key={row.model_name}>
                  <td>{titleCase(row.model_name)}</td>
                  <td>{row.included_in_ensemble ? "Yes" : "No"}</td>
                  <td>{formatNumber(row.weight, 3)}</td>
                  <td>{row.feature_count}</td>
                </tr>
              ))}
              {!components.length ? (
                <tr>
                  <td colSpan={4}>No ensemble component snapshot has been published yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export function IntraFamilyTournamentPage() {
  return (
    <Suspense fallback={<TournamentLoading title="Intra-Family Tournament" />}>
      <IntraFamilyContent />
    </Suspense>
  );
}

export function InterFamilyTournamentPage() {
  return (
    <Suspense fallback={<TournamentLoading title="Inter-Family Tournament" />}>
      <InterFamilyContent />
    </Suspense>
  );
}

export function EnsembleSummaryPage() {
  return (
    <Suspense fallback={<TournamentLoading title="Ensemble Summary Table" />}>
      <EnsembleSummaryContent />
    </Suspense>
  );
}
