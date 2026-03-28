"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ALL_LEAGUES, displayLeagueLabel, normalizeLeague, withLeague } from "@/lib/league";
import type { LeagueCode } from "@/lib/league";
import type { ResearchAdminResponse, ResearchPromotionSummary, ResearchRunRow } from "@/lib/types";
import {
  buildAdminCounts,
  buildChampionTimeline,
  describeChampion,
  formatAdminTimestamp,
  latestRunForChampion,
  summarizeRunSummary,
} from "./helpers";
import styles from "./ResearchAdminView.module.css";

const EMPTY_ADMIN_RESPONSE: ResearchAdminResponse = {
  league: "NBA",
  as_of_utc: null,
  champion: null,
  briefs: [],
  runs: [],
  decisions: [],
};

function statusClassName(baseStatus: string): string {
  const status = baseStatus.toLowerCase();
  if (status.includes("promot")) return `${styles.statusBadge} ${styles.statusPromoted}`;
  if (status.includes("reject") || status.includes("fail")) return `${styles.statusBadge} ${styles.statusRejected}`;
  if (status.includes("active")) return `${styles.statusBadge} ${styles.statusActive}`;
  return styles.statusBadge;
}

function decisionStatus(decision: ResearchPromotionSummary): string {
  return decision.promoted ? "Promoted" : "Rejected";
}

function runStatus(run: ResearchRunRow): string {
  return run.auto_promote ? `${run.status} · auto` : run.status;
}

function runArtifactLinks(run: ResearchRunRow) {
  return [
    run.report_path ? { href: run.report_path, label: "report" } : null,
    run.scorecard_path ? { href: run.scorecard_path, label: "scorecard" } : null,
    run.fold_metrics_path ? { href: run.fold_metrics_path, label: "fold metrics" } : null,
    run.promotion_path ? { href: run.promotion_path, label: "promotion" } : null,
  ].filter((value): value is { href: string; label: string } => Boolean(value));
}

function LeagueToggle({ activeLeague }: { activeLeague: LeagueCode }) {
  return (
    <div className="league-toggle-row" aria-label="Research admin league toggle">
      {ALL_LEAGUES.map((league) => (
        <a
          key={league}
          href={withLeague("/research-admin", league)}
          className={`league-toggle-btn ${league === activeLeague ? "active" : ""}`}
        >
          {league}
        </a>
      ))}
    </div>
  );
}

export default function ResearchAdminView() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const [payload, setPayload] = useState<ResearchAdminResponse>({ ...EMPTY_ADMIN_RESPONSE, league });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError("");
      try {
        const response = await fetch(withLeague("/api/research-admin", league), { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Request failed (${response.status}).`);
        }
        const nextPayload = (await response.json()) as ResearchAdminResponse;
        if (!cancelled) {
          setPayload({
            league,
            as_of_utc: nextPayload.as_of_utc ?? null,
            champion: nextPayload.champion ?? null,
            briefs: Array.isArray(nextPayload.briefs) ? nextPayload.briefs : [],
            runs: Array.isArray(nextPayload.runs) ? nextPayload.runs : [],
            decisions: Array.isArray(nextPayload.decisions) ? nextPayload.decisions : [],
          });
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : "Unable to load research admin.");
          setPayload({ ...EMPTY_ADMIN_RESPONSE, league });
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [league]);

  const counts = useMemo(() => buildAdminCounts(payload), [payload]);
  const timeline = useMemo(() => buildChampionTimeline(payload.champion, payload.decisions), [payload.champion, payload.decisions]);
  const championRun = useMemo(() => latestRunForChampion(payload), [payload]);

  return (
    <div className={styles.page}>
      <section className={`card ${styles.hero}`}>
        <p className={styles.eyebrow}>Research Admin</p>
        <h1 className={styles.headline}>{displayLeagueLabel(league)} Champion Control Room</h1>
        <p className={styles.subcopy}>
          Live-only surface for structured briefs, experiment runs, promotion outcomes, and the champion handoff
          history. Staging keeps the public research desk, but this page stays attached to local SQLite state.
        </p>
        <LeagueToggle activeLeague={league} />
        <div className={styles.kpiGrid}>
          <div className={styles.kpiCard}>
            <p className={styles.kpiLabel}>Active champion</p>
            <p className={styles.kpiValue}>{payload.champion?.model_name ?? "None yet"}</p>
          </div>
          <div className={styles.kpiCard}>
            <p className={styles.kpiLabel}>Briefs tracked</p>
            <p className={styles.kpiValue}>{counts.briefs}</p>
          </div>
          <div className={styles.kpiCard}>
            <p className={styles.kpiLabel}>Runs logged</p>
            <p className={styles.kpiValue}>{counts.runs}</p>
          </div>
          <div className={styles.kpiCard}>
            <p className={styles.kpiLabel}>Promotions approved</p>
            <p className={styles.kpiValue}>{counts.promotions}</p>
          </div>
        </div>
        <div className={styles.metaGrid}>
          <div className={styles.metaBlock}>
            <span className={styles.metaLabel}>Snapshot as of</span>
            <span className={styles.metaValue}>{formatAdminTimestamp(payload.as_of_utc)}</span>
          </div>
          <div className={styles.metaBlock}>
            <span className={styles.metaLabel}>Champion context</span>
            <span className={styles.metaValue}>{describeChampion(payload.champion)}</span>
          </div>
          <div className={styles.metaBlock}>
            <span className={styles.metaLabel}>Champion run summary</span>
            <span className={styles.metaValue}>{summarizeRunSummary(championRun?.summary)}</span>
          </div>
        </div>
      </section>

      {isLoading ? <p className="small">Loading research admin...</p> : null}
      {error ? <div className="card">{error}</div> : null}

      {!isLoading && !error ? (
        <div className="grid two">
          <section className="card">
            <div className={styles.sectionHeader}>
              <div>
                <h2 className="title">Champion Timeline</h2>
                <p className="small">Every promotion that survived the machine-checkable gate.</p>
              </div>
            </div>
            {timeline.length ? (
              <div className={styles.timeline}>
                {timeline.map((event) => (
                  <article key={event.key} className={styles.timelineItem}>
                    <div className={styles.timelineTop}>
                      <div className={styles.stack}>
                        <p className={styles.timelineModel}>{event.model_name}</p>
                        <span className={statusClassName(event.label)}>{event.label}</span>
                      </div>
                      <span className={styles.timelineDate}>{formatAdminTimestamp(event.occurred_at_utc)}</span>
                    </div>
                    <p className={styles.timelineCopy}>
                      {event.reason_summary || "Promoted into the live desk after clearing the active policy gates."}
                    </p>
                    <p className={styles.timelineCopy}>
                      Run: <code>{event.source_run_id ?? "—"}</code> · Brief: <code>{event.source_brief_id ?? "—"}</code>
                    </p>
                  </article>
                ))}
              </div>
            ) : (
              <p className={styles.empty}>No promotion history has been written yet for this league.</p>
            )}
          </section>

          <section className="card">
            <div className={styles.sectionHeader}>
              <div>
                <h2 className="title">Promotion Decisions</h2>
                <p className="small">Recent accept/reject outcomes from the promotion gate.</p>
              </div>
            </div>
            {payload.decisions.length ? (
              <div className={styles.tableWrap}>
                <table>
                  <thead>
                    <tr>
                      <th>When</th>
                      <th>Verdict</th>
                      <th>Candidate</th>
                      <th>Incumbent</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.decisions.map((decision, index) => (
                      <tr key={`${decision.candidate_model_name || "candidate"}-${decision.created_at_utc || index}`}>
                        <td>{formatAdminTimestamp(decision.created_at_utc)}</td>
                        <td>
                          <span className={statusClassName(decisionStatus(decision))}>{decisionStatus(decision)}</span>
                        </td>
                        <td className={styles.tableCellCode}>{decision.candidate_model_name ?? "—"}</td>
                        <td className={styles.tableCellCode}>{decision.incumbent_model_name ?? "—"}</td>
                        <td>{decision.reason_summary ?? "Machine-readable policy evidence stored."}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className={styles.empty}>No promotion decisions recorded yet.</p>
            )}
          </section>

          <section className="card">
            <div className={styles.sectionHeader}>
              <div>
                <h2 className="title">Experiment Runs</h2>
                <p className="small">Each orchestrated run and the artifacts it produced.</p>
              </div>
            </div>
            {payload.runs.length ? (
              <div className={styles.tableWrap}>
                <table>
                  <thead>
                    <tr>
                      <th>Started</th>
                      <th>Run</th>
                      <th>Brief</th>
                      <th>Matchup</th>
                      <th>Status</th>
                      <th>Summary</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.runs.map((run) => (
                      <tr key={run.run_id}>
                        <td>{formatAdminTimestamp(run.started_at_utc)}</td>
                        <td className={styles.tableCellCode}>{run.run_id}</td>
                        <td className={styles.tableCellCode}>{run.brief_key ?? run.brief_id ?? "—"}</td>
                        <td>
                          <div className={styles.stack}>
                            <span className={styles.tableCellCode}>{run.incumbent_model_name ?? "—"}</span>
                            <span className={styles.tableCellMuted}>vs {run.candidate_model_name ?? "—"}</span>
                          </div>
                        </td>
                        <td>
                          <div className={styles.stack}>
                            <span className={statusClassName(runStatus(run))}>{runStatus(run)}</span>
                            {run.completed_at_utc ? (
                              <span className={styles.tableCellMuted}>Completed {formatAdminTimestamp(run.completed_at_utc)}</span>
                            ) : null}
                          </div>
                        </td>
                        <td>
                          <div className={styles.stack}>
                            <span className={styles.summary}>{summarizeRunSummary(run.summary)}</span>
                            {runArtifactLinks(run).length ? (
                              <div className={styles.linkRow}>
                                {runArtifactLinks(run).map((artifact) => (
                                  <a
                                    key={`${run.run_id}-${artifact.label}`}
                                    className={styles.linkChip}
                                    href={artifact.href}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    {artifact.label}
                                  </a>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className={styles.empty}>
                No experiment runs are stored yet{league !== "NBA" ? " for this league" : ""}. NBA is the only v1
                auto-promotion path.
              </p>
            )}
          </section>

          <section className="card">
            <div className={styles.sectionHeader}>
              <div>
                <h2 className="title">Brief History</h2>
                <p className="small">Structured experiment briefs tracked in SQLite from the authored brief files.</p>
              </div>
            </div>
            {payload.briefs.length ? (
              <div className={styles.tableWrap}>
                <table>
                  <thead>
                    <tr>
                      <th>Updated</th>
                      <th>Brief key</th>
                      <th>Title</th>
                      <th>Status</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.briefs.map((brief) => (
                      <tr key={brief.brief_id}>
                        <td>{formatAdminTimestamp(brief.updated_at_utc)}</td>
                        <td className={styles.tableCellCode}>{brief.brief_key}</td>
                        <td>{brief.title}</td>
                        <td>
                          <span className={statusClassName(brief.status)}>{brief.status}</span>
                        </td>
                        <td className={styles.tableCellCode}>{brief.source_path ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className={styles.empty}>No structured briefs have been persisted yet.</p>
            )}
          </section>
        </div>
      ) : null}
    </div>
  );
}
