"use client";

import { useEffect, useMemo, useState } from "react";
import { formatUsd } from "@/lib/currency";
import type { LeagueCode } from "@/lib/league";
import { withLeague } from "@/lib/league";
import type { ResearchDeskResponse, TableRow } from "@/lib/types";
import { EMPTY_RESEARCH_DESK } from "./defaults";
import { displayModelName, gateChips, promotionSummary } from "./copy";
import EvidenceStatusCard from "./EvidenceStatusCard";
import styles from "./ResearchDeskExperience.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const STAGING_ASSET_VERSION = process.env.NEXT_PUBLIC_STAGING_ASSET_VERSION || "";
const STATIC_STAGING = process.env.NEXT_PUBLIC_STATIC_STAGING === "1";

function championPolicyValue(policy: TableRow | null | undefined, key: string): unknown {
  if (!policy || typeof policy !== "object" || Array.isArray(policy)) {
    return null;
  }
  if (key in policy) {
    return policy[key];
  }
  const nestedPolicy = policy.policy;
  if (nestedPolicy && typeof nestedPolicy === "object" && !Array.isArray(nestedPolicy)) {
    return (nestedPolicy as TableRow)[key];
  }
  return null;
}

function buildResearchDeskUrl(league: string): string {
  if (!STATIC_STAGING) {
    return withLeague("/api/research-desk", league as LeagueCode);
  }

  const path = `${BASE_PATH}/staging-data/${league.toLowerCase()}/research-desk.json`;
  return STAGING_ASSET_VERSION ? `${path}?v=${encodeURIComponent(STAGING_ASSET_VERSION)}` : path;
}

function formatAsOf(value?: string | null): string {
  if (!value) return "Unknown";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatTipTime(value?: string | null): string {
  if (!value) return "Time TBD";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Chicago",
  });
}

function formatMoneyline(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return "—";
  const rounded = Math.round(value);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

function formatPoints(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pts`;
}

function formatProbability(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function buildSlateSummary(data: ResearchDeskResponse): string {
  const games = data.counts.total_games;
  const bets = data.counts.bets;
  const passes = data.counts.passes;
  if (games <= 0) {
    return "No games are loaded for this review yet.";
  }
  return `Today the model reviewed ${games} games. It found ${bets} possible ${bets === 1 ? "bet" : "bets"} and passed on ${passes} ${passes === 1 ? "game" : "games"}.`;
}

function buildTrustSummary(data: ResearchDeskResponse): string {
  if (data.production_ready) {
    return data.champion?.model_name
      ? `${displayModelName(data.champion.model_name)} is the approved model for this page.`
      : "The current model has passed the required checks for full use.";
  }
  if (data.champion?.model_name) {
    return `${displayModelName(data.champion.model_name)} is the current approved model. New challengers still need to pass more checks before replacing it.`;
  }
  return "No model is fully approved yet, so treat these picks as testing output, not final betting instructions.";
}

function formatRiskLabel(value?: string | null): string {
  return value === "guarded" ? "More cautious than usual" : "Normal";
}

function formatApprovalLabel(data: ResearchDeskResponse): string {
  if (data.production_ready) return "Approved";
  if (data.promotion_eligible) return "Ready for review";
  return "Still being tested";
}

function cleanDecisionReason(value?: string | null): string {
  const reason = String(value || "").trim();
  if (!reason) return "No explanation is available yet.";
  const normalized = reason.toLowerCase();
  if (normalized.includes("adjusted price fair")) {
    return "The odds are close to the model’s estimate, so there is no clear advantage.";
  }
  if (normalized.includes("favorite underpriced")) {
    return "The model thinks the favorite has a better chance than the odds suggest.";
  }
  if (normalized.includes("underdog underpriced")) {
    return "The model thinks the underdog has a better chance than the odds suggest.";
  }
  if (normalized.includes("no recommendation")) {
    return "The model did not find a clear pick for this game.";
  }
  return reason;
}

export default function ResearchDeskExperience({ league }: { league: LeagueCode }) {
  const [data, setData] = useState<ResearchDeskResponse>(EMPTY_RESEARCH_DESK);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      setIsLoading(true);
      setError("");
      try {
        const response = await fetch(buildResearchDeskUrl(league), {
          cache: STATIC_STAGING ? "force-cache" : "no-store",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Request failed (${response.status}).`);
        }
        const payload = (await response.json()) as ResearchDeskResponse;
        if (!cancelled) {
          setData(payload);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setData({ ...EMPTY_RESEARCH_DESK, league });
          setError(fetchError instanceof Error ? fetchError.message : "Unable to load the research desk right now.");
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
      controller.abort();
    };
  }, [league]);

  const promotionGates = useMemo(() => gateChips(data.latest_promotion?.policy), [data.latest_promotion?.policy]);
  const championDiagnostics = data.champion?.model_name ? data.model_diagnostics?.[data.champion.model_name] : null;
  const unsupportedLeague = data.league !== league;

  if (error) {
    return <div className={`card ${styles.errorState}`}>{error}</div>;
  }

  if (isLoading) {
    return <p className={styles.loadingState}>Loading the latest research desk...</p>;
  }

  return (
    <div className={styles.page}>
      <section className={`card ${styles.heroCard}`}>
        <div className={styles.heroTop}>
          <div>
            <p className={styles.eyebrow}>Daily review</p>
            <h1 className={styles.title}>Today’s Model Review</h1>
            <p className={styles.copy}>{buildSlateSummary(data)}</p>
            <p className={styles.copy}>{buildTrustSummary(data)}</p>
          </div>
          <div className={styles.heroStatusPanel} aria-label="Daily review status">
            <div>
              <span>Approval</span>
              <strong>{formatApprovalLabel(data)}</strong>
            </div>
            <div>
              <span>Risk level</span>
              <strong>{formatRiskLabel(data.desk_posture)}</strong>
            </div>
            <div>
              <span>Approved model</span>
              <strong>{data.champion?.model_name ? displayModelName(data.champion.model_name) : "None yet"}</strong>
            </div>
          </div>
        </div>

        <div className={styles.countGrid}>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Games reviewed</span>
            <strong className={styles.countValue}>{data.counts.total_games}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Suggested bets</span>
            <strong className={styles.countValue}>{data.counts.bets}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>No-bet games</span>
            <strong className={styles.countValue}>{data.counts.passes}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Updated</span>
            <strong className={styles.factValue}>{formatAsOf(data.as_of_utc || data.odds_as_of_utc)}</strong>
          </div>
        </div>

        {unsupportedLeague ? (
          <div className={styles.unsupportedNote}>
            <p className={styles.smallCopy}>MLB is the shipped lane for this dashboard.</p>
          </div>
        ) : null}
      </section>

      <div className={styles.summaryGrid}>
        {data.champion ? (
          <section className={`card ${styles.summaryCard}`}>
            <div className={styles.sectionHeader}>
              <div>
                <span className={styles.sectionLabel}>Model</span>
                <h2 className={styles.sectionTitle}>Approved model in use</h2>
              </div>
            </div>
            <div className={styles.pillRow}>
              <span className={styles.pill}>{displayModelName(data.champion.model_name)}</span>
              {data.champion.source_run_id ? <span className={styles.pill}>Run {data.champion.source_run_id}</span> : null}
              {data.champion.source_brief_id ? <span className={styles.pill}>Brief {data.champion.source_brief_id}</span> : null}
            </div>
            <div className={styles.countGrid}>
              <div className={styles.factTile}>
                <span className={styles.factLabel}>Approved</span>
                <strong className={styles.factValue}>{formatAsOf(data.champion.promoted_at_utc)}</strong>
              </div>
              <div className={styles.factTile}>
                <span className={styles.factLabel}>Maximum expected loss</span>
                <strong className={styles.factValue}>
                  {typeof championPolicyValue(data.champion.policy, "max_mean_drawdown_dollars") === "number"
                    ? formatUsd(Number(championPolicyValue(data.champion.policy, "max_mean_drawdown_dollars")))
                    : "—"}
                </strong>
              </div>
              <div className={styles.factTile}>
                <span className={styles.factLabel}>Minimum bets in review</span>
                <strong className={styles.factValue}>
                  {typeof championPolicyValue(data.champion.policy, "min_bet_count") === "number"
                    ? String(championPolicyValue(data.champion.policy, "min_bet_count"))
                    : "—"}
                </strong>
              </div>
            </div>
            {championDiagnostics ? (
              <div className={styles.diagnosticPanel}>
                <div className={styles.diagnosticStats}>
                  <span>{championDiagnostics.active_feature_count ?? 0} inputs considered</span>
                  <span>{championDiagnostics.fit_active_feature_count ?? 0} inputs used</span>
                  <span>{championDiagnostics.theory_trace?.governance ?? "dashboard explanation"}</span>
                </div>
              </div>
            ) : null}
          </section>
        ) : null}

        <EvidenceStatusCard
          evidenceStatus={data.evidence_status}
          promotionEligible={data.promotion_eligible}
          productionReady={data.production_ready}
          sourceStatus={data.source_status}
        />

        <section className={`card ${styles.summaryCard}`}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.sectionLabel}>Review</span>
              <h2 className={styles.sectionTitle}>Why the latest model was not approved</h2>
            </div>
          </div>
          {data.latest_promotion ? (
            <>
              {data.latest_promotion.candidate_model_name ? (
                <div className={styles.modelReviewBlock}>
                  <div>
                    <span className={styles.sectionLabel}>Model tested</span>
                    <p className={styles.modelSpotlightTitle}>
                      {displayModelName(data.latest_promotion.candidate_model_name)}
                    </p>
                  </div>
                </div>
              ) : null}
              <div className={styles.statusFacts}>
                <span>
                  Decision: <strong>{data.latest_promotion.promoted ? "Approved" : "Not approved"}</strong>
                </span>
                {data.latest_promotion.candidate_model_name ? (
                  <span>
                    Tested model: <strong>{displayModelName(data.latest_promotion.candidate_model_name)}</strong>
                  </span>
                ) : null}
                {data.latest_promotion.incumbent_model_name ? (
                  <span>
                    Compared with: <strong>{displayModelName(data.latest_promotion.incumbent_model_name)}</strong>
                  </span>
                ) : null}
              </div>
              <p className={styles.copy}>{promotionSummary(data.latest_promotion)}</p>
              {promotionGates.length ? (
                <div className={styles.checkList} aria-label="Model approval checks that still need work">
                  {promotionGates.map((gate) => (
                    <div key={gate.label} className={styles.checkItem}>
                      <span className={gate.passed ? styles.checkStatusPass : styles.checkStatus}>Needs work</span>
                      <div>
                        <strong>{gate.label}</strong>
                        <p>{gate.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
              <p className={styles.smallCopy}>Reviewed {formatAsOf(data.latest_promotion.created_at_utc)}</p>
            </>
          ) : (
            <p className={styles.smallCopy}>No model approval review has been loaded yet.</p>
          )}
        </section>
      </div>

      <section className={`card ${styles.tableCard}`}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.sectionLabel}>Games</span>
            <h2 className={styles.sectionTitle}>Today’s picks and passes</h2>
          </div>
        </div>

        {data.rows.length === 0 ? (
          <div className={styles.emptyState}>No games are loaded yet.</div>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Matchup</th>
                  <th>First pitch</th>
                  <th>Decision</th>
                  <th>Bet size</th>
                  <th>Sportsbook odds</th>
                  <th>Model difference</th>
                  <th>Why</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr key={row.game_id}>
                    <td data-label="Matchup">
                      <div className={styles.matchup}>
                        <span className={styles.matchupTitle}>
                          {row.away_team} at {row.home_team}
                        </span>
                        <span className={styles.metaText}>
                          Model says the home team wins {formatProbability(row.home_win_probability)}
                        </span>
                      </div>
                    </td>
                    <td data-label="First pitch" className={styles.metricMuted}>{formatTipTime(row.start_time_utc)}</td>
                    <td data-label="Decision">
                      <span className={`${styles.statusPill} ${row.bet_label === "bet" ? styles.statusBet : styles.statusPass}`}>
                        {row.bet_label === "bet" ? `Bet ${row.team || ""}`.trim() : "No bet"}
                      </span>
                    </td>
                    <td data-label="Bet size" className={styles.metricStrong}>{row.stake > 0 ? formatUsd(row.stake) : "—"}</td>
                    <td data-label="Sportsbook odds" className={styles.metricMuted}>{formatMoneyline(row.odds)}</td>
                    <td data-label="Model difference" className={styles.metricStrong}>{formatPoints(row.edge)}</td>
                    <td data-label="Why">
                      <div className={styles.reasonCell}>
                        <span className={styles.metricStrong}>{cleanDecisionReason(row.reason)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
