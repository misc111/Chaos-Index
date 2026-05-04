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

function formatExpectedValue(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatProbability(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function titleCaseToken(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function ResearchDeskExperience({ league }: { league: LeagueCode }) {
  const [data, setData] = useState<ResearchDeskResponse>(EMPTY_RESEARCH_DESK);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError("");
      try {
        const response = await fetch(buildResearchDeskUrl(league), {
          cache: STATIC_STAGING ? "force-cache" : "no-store",
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
    };
  }, [league]);

  const promotionGates = useMemo(() => gateChips(data.latest_promotion?.policy), [data.latest_promotion?.policy]);
  const championDiagnostics = data.champion?.model_name ? data.model_diagnostics?.[data.champion.model_name] : null;
  const unsupportedLeague = data.league !== league;
  const evidenceStage = data.evidence_status?.evidence_stage || data.evidence_stage;

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
            <p className={styles.eyebrow}>MLB Desk</p>
            <h1 className={styles.title}>Research Desk</h1>
            <p className={styles.copy}>
              {data.overnight_summary || "Today’s slate, model status, and bet calls."}
            </p>
          </div>
          <div className={styles.heroMeta}>
            <span className={styles.pill}>{data.league}</span>
            <span className={`${styles.pill} ${data.production_ready ? styles.postureNormal : styles.postureGuarded}`}>
              {evidenceStage ? titleCaseToken(evidenceStage) : data.production_ready ? "Production Ready" : "Not Production Ready"}
            </span>
            <span
              className={`${styles.pill} ${
                data.desk_posture === "guarded" ? styles.postureGuarded : styles.postureNormal
              }`}
            >
              {data.desk_posture === "guarded" ? "Guarded risk" : "Normal risk"}
            </span>
            <span className={styles.pill}>{data.champion?.model_name ? displayModelName(data.champion.model_name) : "Fallback model"}</span>
          </div>
        </div>

        <div className={styles.countGrid}>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Games</span>
            <strong className={styles.countValue}>{data.counts.total_games}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Bets</span>
            <strong className={styles.countValue}>{data.counts.bets}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Passes</span>
            <strong className={styles.countValue}>{data.counts.passes}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Updated</span>
            <strong className={styles.factValue}>{formatAsOf(data.as_of_utc || data.odds_as_of_utc)}</strong>
          </div>
        </div>

        {unsupportedLeague ? (
          <div className={styles.unsupportedNote}>
            <p className={styles.smallCopy}>
              MLB is the shipped lane for this dashboard.
            </p>
          </div>
        ) : null}
      </section>

      <div className={styles.summaryGrid}>
        {data.champion ? (
          <section className={`card ${styles.summaryCard}`}>
            <div className={styles.sectionHeader}>
              <div>
                <span className={styles.sectionLabel}>Model</span>
                <h2 className={styles.sectionTitle}>Active Model</h2>
              </div>
            </div>
            <>
              <div className={styles.pillRow}>
                <span className={styles.pill}>{displayModelName(data.champion.model_name)}</span>
                {data.champion.source_run_id ? <span className={styles.pill}>Run {data.champion.source_run_id}</span> : null}
                {data.champion.source_brief_id ? <span className={styles.pill}>Brief {data.champion.source_brief_id}</span> : null}
              </div>
              <div className={styles.countGrid}>
                <div className={styles.factTile}>
                  <span className={styles.factLabel}>Promoted</span>
                  <strong className={styles.factValue}>{formatAsOf(data.champion.promoted_at_utc)}</strong>
                </div>
                <div className={styles.factTile}>
                  <span className={styles.factLabel}>Drawdown cap</span>
                  <strong className={styles.factValue}>
                    {typeof championPolicyValue(data.champion.policy, "max_mean_drawdown_dollars") === "number"
                      ? formatUsd(Number(championPolicyValue(data.champion.policy, "max_mean_drawdown_dollars")))
                      : "—"}
                  </strong>
                </div>
                <div className={styles.factTile}>
                  <span className={styles.factLabel}>Min bet count</span>
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
                    <span>{championDiagnostics.active_feature_count ?? 0} fitted inputs</span>
                    <span>{championDiagnostics.fit_active_feature_count ?? 0} coefficient-active</span>
                    <span>{championDiagnostics.theory_trace?.governance ?? "dashboard/reporting layer"}</span>
                  </div>
                  <div className={styles.featureList}>
                    {(championDiagnostics.active_features || []).map((feature) => (
                      <code key={`${data.champion?.model_name}-${feature}`}>{feature}</code>
                    ))}
                  </div>
                  {data.model_feature_map_run_id || data.model_feature_set_version ? (
                    <p className={styles.smallCopy}>
                      Fit artifact <code>{data.model_feature_map_run_id ?? "—"}</code> · feature set{" "}
                      <code>{data.model_feature_set_version ?? "—"}</code>
                    </p>
                  ) : null}
                </div>
              ) : null}
            </>
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
              <h2 className={styles.sectionTitle}>Promotion Status</h2>
            </div>
          </div>
          {data.latest_promotion ? (
            <>
              <div className={styles.promotionMeta}>
                <span
                  className={`${styles.statusPill} ${
                    data.latest_promotion.promoted ? styles.statusBet : styles.statusPass
                  }`}
                >
                  {data.latest_promotion.promoted ? "Promoted" : "Rejected"}
                </span>
                {data.latest_promotion.candidate_model_name ? (
                  <span className={styles.pill}>{displayModelName(data.latest_promotion.candidate_model_name)}</span>
                ) : null}
                {data.latest_promotion.incumbent_model_name ? (
                  <span className={styles.pill}>vs {displayModelName(data.latest_promotion.incumbent_model_name)}</span>
                ) : null}
              </div>
              <p className={styles.copy}>{promotionSummary(data.latest_promotion)}</p>
              {promotionGates.length ? (
                <div className={styles.gateRow}>
                  {promotionGates.map((gate) => (
                    <span
                      key={gate.label}
                      className={`${styles.gatePill} ${gate.passed ? styles.gatePass : styles.gateFail}`}
                    >
                      {gate.label}
                    </span>
                  ))}
                </div>
              ) : null}
              <p className={styles.smallCopy}>Reviewed {formatAsOf(data.latest_promotion.created_at_utc)}</p>
            </>
          ) : (
            <p className={styles.smallCopy}>No promotion review yet.</p>
          )}
        </section>
      </div>

      <section className={`card ${styles.tableCard}`}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.sectionLabel}>Slate</span>
            <h2 className={styles.sectionTitle}>Today’s Calls</h2>
          </div>
        </div>

        {data.rows.length === 0 ? (
          <div className={styles.emptyState}>No slate loaded.</div>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Matchup</th>
                  <th>Tip</th>
                  <th>Call</th>
                  <th>Stake</th>
                  <th>Odds</th>
                  <th>Edge</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr key={row.game_id}>
                    <td>
                      <div className={styles.matchup}>
                        <span className={styles.matchupTitle}>
                          {row.away_team} at {row.home_team}
                        </span>
                        <span className={styles.metaText}>
                          Home win {formatProbability(row.home_win_probability)}
                          {row.betting_model_name ? ` · ${row.betting_model_name}` : ""}
                        </span>
                      </div>
                    </td>
                    <td className={styles.metricMuted}>{formatTipTime(row.start_time_utc)}</td>
                    <td>
                      <span className={`${styles.statusPill} ${row.bet_label === "bet" ? styles.statusBet : styles.statusPass}`}>
                        {row.bet_label === "bet" ? `Bet ${row.team || ""}`.trim() : "Pass"}
                      </span>
                    </td>
                    <td className={styles.metricStrong}>{row.stake > 0 ? formatUsd(row.stake) : "—"}</td>
                    <td className={styles.metricMuted}>{formatMoneyline(row.odds)}</td>
                    <td className={styles.metricStrong}>{formatPoints(row.edge)}</td>
                    <td>
                      <div className={styles.reasonCell}>
                        <span className={styles.metricStrong}>{row.reason}</span>
                        {typeof row.expected_value === "number" && Number.isFinite(row.expected_value) ? (
                          <span className={styles.metaText}>EV {formatExpectedValue(row.expected_value)}</span>
                        ) : null}
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
