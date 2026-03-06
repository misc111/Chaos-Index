"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { computeBetDecision } from "@/lib/betting";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";
import { type MarketBoardResponse, type MarketBoardRow } from "@/lib/types";
import styles from "./styles.module.css";

type DerivedBoardRow = {
  row: MarketBoardRow;
  modelWinner: string;
  modelWinnerProbability: number;
  fairHomeMoneyline: number | null;
  fairAwayMoneyline: number | null;
  fairWinnerMoneyline: number | null;
  bestSignalLabel: string;
  bestSignalReason: string;
  bestSignalClass: string;
  edgeLabel: string;
  valueTeam: string | null;
  valueOdds: number | null;
  edgeValue: number | null;
};

function formatAsOfLabel(value?: string | null): string {
  if (!value) return "No snapshot";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatCentralTip(value?: string | null): string {
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
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric === 0) return "—";
  const rounded = Math.round(numeric);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

function formatSpreadPoint(value?: number | null, isHome?: boolean): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  const oriented = isHome ? -numeric : numeric;
  return oriented > 0 ? `+${oriented.toFixed(1)}` : oriented.toFixed(1);
}

function formatTotalPoint(value?: number | null): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return numeric.toFixed(1);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function probabilityToAmericanOdds(probability: number): number | null {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) return null;
  if (probability === 0.5) return 100;
  return probability >= 0.5 ? (-100 * probability) / (1 - probability) : (100 * (1 - probability)) / probability;
}

function booksLabel(count?: number): string {
  const numeric = Number(count);
  if (!Number.isFinite(numeric) || numeric <= 0) return "No books";
  return `${numeric} ${numeric === 1 ? "book" : "books"}`;
}

function buildDerivedRow(row: MarketBoardRow): DerivedBoardRow {
  const modelWinnerIsHome = row.home_win_probability >= 0.5;
  const modelWinner = modelWinnerIsHome ? row.home_team_name : row.away_team_name;
  const modelWinnerProbability = modelWinnerIsHome ? row.home_win_probability : 1 - row.home_win_probability;
  const fairHomeMoneyline = probabilityToAmericanOdds(row.home_win_probability);
  const fairAwayMoneyline = probabilityToAmericanOdds(1 - row.home_win_probability);
  const fairWinnerMoneyline = modelWinnerIsHome ? fairHomeMoneyline : fairAwayMoneyline;

  const decision = computeBetDecision({
    home_team: row.home_team_name,
    away_team: row.away_team_name,
    home_win_probability: row.home_win_probability,
    home_moneyline: row.moneyline.home_price,
    away_moneyline: row.moneyline.away_price,
  });

  const edgeValue = typeof decision.edge === "number" && Number.isFinite(decision.edge) ? decision.edge : null;
  const edgeLabel = edgeValue === null ? "No edge" : `${edgeValue > 0 ? "+" : ""}${(edgeValue * 100).toFixed(1)} pts`;
  const valueTeam = decision.stake > 0 ? decision.team : null;
  const valueOdds = decision.stake > 0 && typeof decision.odds === "number" ? decision.odds : null;

  if (decision.stake > 0 && decision.team) {
    return {
      row,
      modelWinner,
      modelWinnerProbability,
      fairHomeMoneyline,
      fairAwayMoneyline,
      fairWinnerMoneyline,
      bestSignalLabel: `Value on ${decision.team}`,
      bestSignalReason: `${decision.reason} · ${edgeLabel}`,
      bestSignalClass: styles.signalPositive,
      edgeLabel,
      valueTeam,
      valueOdds,
      edgeValue,
    };
  }

  return {
    row,
    modelWinner,
    modelWinnerProbability,
    fairHomeMoneyline,
    fairAwayMoneyline,
    fairWinnerMoneyline,
    bestSignalLabel: `Likeliest winner: ${modelWinner}`,
    bestSignalReason: `Fair ML ${formatMoneyline(fairWinnerMoneyline)} · ${decision.reason}`,
    bestSignalClass: styles.signalNeutral,
    edgeLabel,
    valueTeam: null,
    valueOdds: null,
    edgeValue,
  };
}

function MarketBoardPageContent() {
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));
  const [report, setReport] = useState<MarketBoardResponse>({
    league,
    as_of_utc: null,
    odds_as_of_utc: null,
    date_central: undefined,
    rows: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadReport() {
      setLoading(true);
      setError("");

      try {
        const payload = await fetchDashboardJson<Partial<MarketBoardResponse>>("marketBoard", "/api/market-board", league);

        if (!cancelled) {
          setReport({
            league: String(payload.league || league),
            as_of_utc: payload.as_of_utc ?? null,
            odds_as_of_utc: payload.odds_as_of_utc ?? null,
            date_central: payload.date_central,
            rows: Array.isArray(payload.rows) ? payload.rows : [],
          });
        }
      } catch (fetchError) {
        if (!cancelled) {
          const message =
            fetchError instanceof Error ? fetchError.message : "Unable to load the market board right now.";
          setError(message);
          setReport({
            league,
            as_of_utc: null,
            odds_as_of_utc: null,
            date_central: undefined,
            rows: [],
          });
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadReport();

    return () => {
      cancelled = true;
    };
  }, [league]);

  const derivedRows = useMemo(() => report.rows.map((row) => buildDerivedRow(row)), [report.rows]);
  const strongestLean = useMemo(() => {
    if (!derivedRows.length) return null;
    return derivedRows.reduce((best, current) =>
      current.modelWinnerProbability > best.modelWinnerProbability ? current : best
    );
  }, [derivedRows]);
  const bestValue = useMemo(() => {
    const candidates = derivedRows.filter((entry) => typeof entry.edgeValue === "number" && entry.edgeValue > 0);
    if (!candidates.length) return null;
    return candidates.reduce((best, current) => ((current.edgeValue || 0) > (best.edgeValue || 0) ? current : best));
  }, [derivedRows]);

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroLead}>
          <p className={styles.eyebrow}>{league} sportsbook-style scan</p>
          <h2 className={styles.headline}>Market Board</h2>
          <p className={styles.description}>
            A board-first view of today&apos;s slate. Spread, total, and moneyline mirror the book layout, while the model strip
            adds fair price and value context that the sportsbook does not.
          </p>
          <div className={styles.heroNotes}>
            <span className={styles.heroNote}>Today only</span>
            <span className={styles.heroNote}>Consensus line, best current price</span>
            <span className={styles.heroNote}>Model overlay built from ensemble win probability</span>
          </div>
        </div>

        <div className={styles.summaryGrid}>
          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Forecast snapshot</p>
            <p className={styles.summaryValue}>{formatAsOfLabel(report.as_of_utc)}</p>
            <p className={styles.summaryHint}>Latest probability snapshot feeding the model overlay.</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Odds snapshot</p>
            <p className={styles.summaryValue}>{formatAsOfLabel(report.odds_as_of_utc)}</p>
            <p className={styles.summaryHint}>Current board lines pulled from the latest odds snapshot.</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Games on board</p>
            <p className={styles.summaryValue}>{derivedRows.length}</p>
            <p className={styles.summaryHint}>Only games scheduled today in Central Time are shown.</p>
          </article>

          <article className={styles.summaryCard}>
            <p className={styles.summaryLabel}>Best current signal</p>
            <p className={styles.summaryValue}>
              {bestValue ? bestValue.bestSignalLabel : strongestLean ? strongestLean.modelWinner : "No signal"}
            </p>
            <p className={styles.summaryHint}>
              {bestValue
                ? `${bestValue.bestSignalReason}${bestValue.valueOdds ? ` · ${formatMoneyline(bestValue.valueOdds)}` : ""}`
                : strongestLean
                  ? `${formatPercent(strongestLean.modelWinnerProbability)} model win chance`
                  : "No games are available in the current snapshot."}
            </p>
          </article>
        </div>
      </section>

      <section className={styles.boardPanel}>
        <div className={styles.boardIntro}>
          <div>
            <p className={styles.boardEyebrow}>Board view</p>
            <h3 className={styles.boardTitle}>Today&apos;s market board</h3>
          </div>
          <p className={styles.boardHint}>
            Spread and total show the most common line across books. Prices show the best current number at that line.
          </p>
        </div>

        <div className={styles.boardHeader} aria-hidden="true">
          <span>Matchup</span>
          <span>Spread</span>
          <span>Total</span>
          <span>Money</span>
          <span>Model Signal</span>
        </div>

        {loading ? <p className={styles.boardStatus}>Loading market board...</p> : null}
        {error ? <p className={styles.boardError}>Failed to load market board: {error}</p> : null}
        {!loading && !error && !derivedRows.length ? <p className={styles.boardStatus}>No games scheduled today.</p> : null}

        {!loading && !error && derivedRows.length ? (
          <div className={styles.boardList}>
            {derivedRows.map((entry) => {
              const { row } = entry;
              const tipLabel = row.start_time_utc ? `${formatCentralTip(row.start_time_utc)} CT` : "Time TBD";
              return (
                <article key={row.game_id} className={styles.boardRow}>
                  <div className={styles.matchupCell}>
                    <div className={styles.matchMeta}>
                      <span className={styles.tipBadge}>{tipLabel}</span>
                      <span className={styles.booksMeta}>{booksLabel(row.moneyline.books_count)}</span>
                    </div>

                    <div className={styles.teamsStack}>
                      <div className={styles.teamLine}>
                        <span className={styles.teamName}>{row.away_team_name}</span>
                        <span className={styles.teamCode}>{row.away_team}</span>
                      </div>
                      <div className={styles.teamLine}>
                        <span className={styles.teamName}>{row.home_team_name}</span>
                        <span className={styles.teamCode}>{row.home_team}</span>
                      </div>
                    </div>
                  </div>

                  <div className={styles.marketCell}>
                    <p className={styles.marketLabel}>Spread</p>
                    <div className={styles.quoteRow}>
                      <span>{formatSpreadPoint(row.spread.point, false)}</span>
                      <strong>{formatMoneyline(row.spread.away_price)}</strong>
                    </div>
                    <div className={styles.quoteRow}>
                      <span>{formatSpreadPoint(row.spread.point, true)}</span>
                      <strong>{formatMoneyline(row.spread.home_price)}</strong>
                    </div>
                    <p className={styles.marketMeta}>{booksLabel(row.spread.books_count)}</p>
                  </div>

                  <div className={styles.marketCell}>
                    <p className={styles.marketLabel}>Total</p>
                    <div className={styles.quoteRow}>
                      <span>O {formatTotalPoint(row.total.point)}</span>
                      <strong>{formatMoneyline(row.total.over_price)}</strong>
                    </div>
                    <div className={styles.quoteRow}>
                      <span>U {formatTotalPoint(row.total.point)}</span>
                      <strong>{formatMoneyline(row.total.under_price)}</strong>
                    </div>
                    <p className={styles.marketMeta}>{booksLabel(row.total.books_count)}</p>
                  </div>

                  <div className={styles.marketCell}>
                    <p className={styles.marketLabel}>Money</p>
                    <div className={styles.quoteRow}>
                      <span>{row.away_team}</span>
                      <strong>{formatMoneyline(row.moneyline.away_price)}</strong>
                    </div>
                    <div className={styles.quoteRow}>
                      <span>{row.home_team}</span>
                      <strong>{formatMoneyline(row.moneyline.home_price)}</strong>
                    </div>
                    <p className={styles.marketMeta}>Best current moneyline by side</p>
                  </div>

                  <div className={styles.signalCell}>
                    <p className={styles.marketLabel}>Model signal</p>
                    <div className={`${styles.signalBadge} ${entry.bestSignalClass}`}>{entry.bestSignalLabel}</div>
                    <p className={styles.signalText}>
                      {entry.modelWinner} {formatPercent(entry.modelWinnerProbability)} likely winner
                    </p>
                    <p className={styles.signalText}>
                      Fair ML H {formatMoneyline(entry.fairHomeMoneyline)} · A {formatMoneyline(entry.fairAwayMoneyline)}
                    </p>
                    <p className={styles.signalText}>{entry.bestSignalReason}</p>
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}
      </section>
    </div>
  );
}

export default function MarketBoardPage() {
  return (
    <Suspense fallback={<p className="small">Loading market board...</p>}>
      <MarketBoardPageContent />
    </Suspense>
  );
}
