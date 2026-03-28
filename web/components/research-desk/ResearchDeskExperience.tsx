"use client";

import { useEffect, useMemo, useState } from "react";
import { formatUsd } from "@/lib/currency";
import type { LeagueCode } from "@/lib/league";
import { withLeague } from "@/lib/league";
import type { ResearchDeskResponse, TableRow } from "@/lib/types";
import styles from "./ResearchDeskExperience.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const STAGING_ASSET_VERSION = process.env.NEXT_PUBLIC_STAGING_ASSET_VERSION || "";
const STATIC_STAGING = process.env.NEXT_PUBLIC_STATIC_STAGING === "1";

const EMPTY_RESEARCH_DESK: ResearchDeskResponse = {
  league: "NBA",
  as_of_utc: null,
  odds_as_of_utc: null,
  date_central: undefined,
  desk_posture: "normal",
  overnight_summary: null,
  champion: null,
  latest_promotion: null,
  counts: {
    total_games: 0,
    bets: 0,
    passes: 0,
  },
  rows: [],
};

type GateChip = {
  label: string;
  passed: boolean;
};

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
    return withLeague("/api/research-desk", league as "NBA" | "NHL" | "NCAAM");
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
  return value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function gateChips(policy?: TableRow | null): GateChip[] {
  const gates = policy?.gates;
  if (!gates || typeof gates !== "object" || Array.isArray(gates)) {
    return [];
  }

  return Object.entries(gates)
    .filter(([, value]) => typeof value === "boolean")
    .map(([key, value]) => ({
      label: titleCaseToken(key),
      passed: Boolean(value),
    }));
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
  const unsupportedLeague = data.league !== "NBA";

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
            <p className={styles.eyebrow}>Research Desk</p>
            <h1 className={styles.title}>Nightly underwriting sheet</h1>
            <p className={styles.copy}>
              {data.overnight_summary || "The desk summarizes champion state, risk posture, and tonight's bet or pass calls."}
            </p>
          </div>
          <div className={styles.heroMeta}>
            <span className={styles.pill}>League {data.league}</span>
            <span
              className={`${styles.pill} ${
                data.desk_posture === "guarded" ? styles.postureGuarded : styles.postureNormal
              }`}
            >
              {data.desk_posture === "guarded" ? "Guarded posture" : "Normal posture"}
            </span>
            <span className={styles.pill}>Model {data.champion?.model_name || "fallback"}</span>
          </div>
        </div>

        <div className={styles.countGrid}>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Games on desk</span>
            <strong className={styles.countValue}>{data.counts.total_games}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Bets tonight</span>
            <strong className={styles.countValue}>{data.counts.bets}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>Passes tonight</span>
            <strong className={styles.countValue}>{data.counts.passes}</strong>
          </div>
          <div className={styles.countTile}>
            <span className={styles.countLabel}>As of</span>
            <strong className={styles.factValue}>{formatAsOf(data.as_of_utc || data.odds_as_of_utc)}</strong>
          </div>
        </div>

        {unsupportedLeague ? (
          <div className={styles.unsupportedNote}>
            <p className={styles.smallCopy}>
              This view is intentionally NBA-first in v1. Other leagues continue to use the existing dashboard surfaces until the
              promotion loop is widened.
            </p>
          </div>
        ) : null}
      </section>

      <div className={styles.summaryGrid}>
        <section className={`card ${styles.summaryCard}`}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.sectionLabel}>Champion</span>
              <h2 className={styles.sectionTitle}>Active model summary</h2>
            </div>
          </div>
          {data.champion ? (
            <>
              <div className={styles.pillRow}>
                <span className={styles.pill}>{data.champion.model_name}</span>
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
            </>
          ) : (
            <p className={styles.smallCopy}>No promoted champion has been recorded yet.</p>
          )}
        </section>

        <section className={`card ${styles.summaryCard}`}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.sectionLabel}>Promotion</span>
              <h2 className={styles.sectionTitle}>Latest rationale</h2>
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
                  <span className={styles.pill}>Candidate {data.latest_promotion.candidate_model_name}</span>
                ) : null}
                {data.latest_promotion.incumbent_model_name ? (
                  <span className={styles.pill}>Incumbent {data.latest_promotion.incumbent_model_name}</span>
                ) : null}
              </div>
              <p className={styles.copy}>{data.latest_promotion.reason_summary || "No summary was stored for the last promotion decision."}</p>
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
              <p className={styles.smallCopy}>Decision time {formatAsOf(data.latest_promotion.created_at_utc)}</p>
            </>
          ) : (
            <p className={styles.smallCopy}>No promotion decision has been written yet.</p>
          )}
        </section>
      </div>

      <section className={`card ${styles.tableCard}`}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.sectionLabel}>Tonight</span>
            <h2 className={styles.sectionTitle}>Bet, pass, size, reason</h2>
          </div>
          <p className={styles.smallCopy}>
            The desk rows come from the existing market-board pricing surface and betting rules, not a parallel recommendation stack.
          </p>
        </div>

        {data.rows.length === 0 ? (
          <div className={styles.emptyState}>No games are on the current desk slate.</div>
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
                        <span className={styles.metaText}>EV {formatExpectedValue(row.expected_value)}</span>
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
