"use client";

import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { TeamMatchup } from "@/components/TeamWithIcon";
import { useDashboardData } from "@/lib/hooks/useDashboardData";
import { useLeague } from "@/lib/hooks/useLeague";
import type { ActualVsExpectedResponse } from "@/lib/types";

type CalendarItem = {
  id: string;
  primaryTeam: string;
  secondaryTeam: string;
  status: string;
  detail?: string;
  dotClass: "dot-correct" | "dot-incorrect" | "dot-upcoming" | "dot-tossup";
  kind: "historical" | "upcoming";
  resultClass?: "correct" | "incorrect" | "tossup";
};

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function toDateKey(year: number, monthIndex: number, day: number): string {
  return `${year}-${pad2(monthIndex + 1)}-${pad2(day)}`;
}

function dateKeyFromLocalToday(): string {
  const now = new Date();
  return toDateKey(now.getFullYear(), now.getMonth(), now.getDate());
}

function normalizeDateKey(value: string): string {
  return value.slice(0, 10);
}

function centralDateKeyFromTimestamp(value?: string | null): string | null {
  if (!value) return null;
  const normalized = normalizeUtcTimestamp(value);
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return null;

  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(parsed);

  const year = parts.find((p) => p.type === "year")?.value;
  const month = parts.find((p) => p.type === "month")?.value;
  const day = parts.find((p) => p.type === "day")?.value;
  if (!year || !month || !day) return null;
  return `${year}-${month}-${day}`;
}

function resolveCalendarDateKey(gameDateUtc: string, ...timestamps: Array<string | null | undefined>): string {
  for (const value of timestamps) {
    const localKey = centralDateKeyFromTimestamp(value);
    if (localKey) return localKey;
  }
  return normalizeDateKey(gameDateUtc);
}

function formatAsOfLabel(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function modeledWinPctLabel(
  probHomeWin: number,
  homeTeam: string,
  awayTeam: string,
  predictedWinner?: string | null
): string {
  const pHome = Number.isFinite(probHomeWin) ? Math.min(1, Math.max(0, probHomeWin)) : 0.5;
  const winner = (predictedWinner || "").trim();
  const resolvedWinner =
    winner === homeTeam || winner === awayTeam ? winner : pHome >= 0.5 ? homeTeam : awayTeam;
  const winProb = resolvedWinner === homeTeam ? pHome : 1 - pHome;
  return `${(winProb * 100).toFixed(1)}%`;
}

function monthTitle(year: number, monthIndex: number): string {
  return new Date(year, monthIndex, 1).toLocaleDateString(undefined, {
    month: "long",
    year: "numeric",
  });
}

function normalizeUtcTimestamp(value: string): string {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$/.test(value)) {
    return value.replace("Z", ":00Z");
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) {
    return `${value}:00Z`;
  }
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(value)) {
    return `${value}Z`;
  }
  return value;
}

function daySuccessRateLabel(items: CalendarItem[] = []): string | null {
  const historical = items.filter((item) => item.kind === "historical");
  if (!historical.length) return null;

  let correct = 0;
  let incorrect = 0;
  for (const item of historical) {
    if (item.resultClass === "correct") correct += 1;
    if (item.resultClass === "incorrect") incorrect += 1;
  }

  const denom = correct + incorrect;
  const pct = denom > 0 ? `${((correct / denom) * 100).toFixed(0)}%` : "—";
  return `Success Rate: ${pct}`;
}

const EMPTY_ACTUAL_VS_EXPECTED: ActualVsExpectedResponse = {
  historical_rows: [],
  upcoming_rows: [],
};

function ActualVsExpectedPageContent() {
  const league = useLeague();
  const searchParams = useSearchParams();
  const refreshNonce = searchParams.get("refreshNonce");
  const { data, isLoading: loading, error } = useDashboardData<ActualVsExpectedResponse>(
    "actualVsExpected",
    "/api/actual-vs-expected",
    league,
    EMPTY_ACTUAL_VS_EXPECTED,
    refreshNonce
  );
  const historicalRows = useMemo(() => data.historical_rows || [], [data.historical_rows]);
  const upcomingRows = useMemo(() => data.upcoming_rows || [], [data.upcoming_rows]);
  const latestAsOf = data.as_of_utc || "";

  const [monthCursor, setMonthCursor] = useState(() => {
    const now = new Date();
    return { year: now.getFullYear(), month: now.getMonth() };
  });

  const todayKey = useMemo(() => dateKeyFromLocalToday(), []);

  const eventsByDate = useMemo(() => {
    const map: Record<string, CalendarItem[]> = {};

    for (const row of historicalRows) {
      const key = resolveCalendarDateKey(row.game_date_utc, row.start_time_utc, row.final_utc);
      if (!map[key]) map[key] = [];

      const isTossUp = Number(row.is_toss_up) === 1;
      const isCorrect = Number(row.model_correct) === 1;
      map[key].push({
        id: `hist-${row.game_id}`,
        primaryTeam: row.home_team,
        secondaryTeam: row.away_team,
        status: isTossUp ? "Near-Even Call" : isCorrect ? "Model Correct" : "Model Incorrect",
        detail: `Modeled win %: ${modeledWinPctLabel(
          Number(row.prob_home_win),
          row.home_team,
          row.away_team,
          row.predicted_winner
        )}`,
        dotClass: isTossUp ? "dot-tossup" : isCorrect ? "dot-correct" : "dot-incorrect",
        kind: "historical",
        resultClass: isTossUp ? "tossup" : isCorrect ? "correct" : "incorrect",
      });
    }

    for (const row of upcomingRows) {
      const key = resolveCalendarDateKey(row.game_date_utc, row.start_time_utc);
      if (!map[key]) map[key] = [];

      const homeWinProb = Number(row.ensemble_prob_home_win);

      map[key].push({
        id: `future-${row.game_id}`,
        primaryTeam: row.home_team,
        secondaryTeam: row.away_team,
        status: `${(homeWinProb * 100).toFixed(1)}%`,
        dotClass: homeWinProb >= 0.5 ? "dot-correct" : "dot-incorrect",
        kind: "upcoming",
      });
    }

    return map;
  }, [historicalRows, upcomingRows]);

  const { year, month } = monthCursor;
  const firstWeekday = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const cells = useMemo(() => {
    const out: Array<{ key: string | null; day?: number; items?: CalendarItem[] }> = [];
    for (let i = 0; i < firstWeekday; i += 1) {
      out.push({ key: null });
    }
    for (let day = 1; day <= daysInMonth; day += 1) {
      const key = toDateKey(year, month, day);
      out.push({ key, day, items: eventsByDate[key] || [] });
    }
    return out;
  }, [daysInMonth, eventsByDate, firstWeekday, month, year]);

  const goToPreviousMonth = () => {
    setMonthCursor((prev) => {
      if (prev.month === 0) return { year: prev.year - 1, month: 11 };
      return { year: prev.year, month: prev.month - 1 };
    });
  };

  const goToNextMonth = () => {
    setMonthCursor((prev) => {
      if (prev.month === 11) return { year: prev.year + 1, month: 0 };
      return { year: prev.year, month: prev.month + 1 };
    });
  };

  return (
    <div className="grid">
      <div className="card">
        <div className="calendar-top-row">
          <h2 className="title">Actual vs Expected</h2>
          <div className="calendar-nav">
            <button type="button" onClick={goToPreviousMonth}>
              Previous
            </button>
            <div className="calendar-month-label">{monthTitle(year, month)}</div>
            <button type="button" onClick={goToNextMonth}>
              Next
            </button>
          </div>
        </div>

        <p className="small">
          Past game markers use the historical ensemble snapshot timestamp recorded before each game finalized.
        </p>
        <p className="small">Near-even markers are a raw prediction diagnostic only. Betting now uses uncertainty-adjusted edge instead of a hard 45%-55% band.</p>
        {latestAsOf ? <p className="small">Upcoming snapshot as of {formatAsOfLabel(latestAsOf)}</p> : null}
        {loading ? <p className="small">Loading calendar...</p> : null}
        {error ? <p className="small">Failed to load: {error}</p> : null}

        <div className="calendar-scroll">
          <div className="calendar-weekdays">
            {WEEKDAYS.map((weekday) => (
              <div key={weekday} className="calendar-weekday">
                {weekday}
              </div>
            ))}
          </div>

          <div className="calendar-grid">
            {cells.map((cell, idx) => {
              if (!cell.key || !cell.day) {
                return <div key={`empty-${idx}`} className="calendar-day calendar-day-empty" />;
              }

              const dayClass =
                cell.key === todayKey
                  ? "calendar-day-today"
                  : cell.key < todayKey
                    ? "calendar-day-past"
                    : "calendar-day-future";
              const hasItems = Boolean(cell.items && cell.items.length > 0);
              const successRateLabel = daySuccessRateLabel(cell.items || []);
              const dayLabel = new Date(year, month, cell.day).toLocaleDateString(undefined, {
                weekday: "short",
                month: "short",
                day: "numeric",
              });

              return (
                <div
                  key={cell.key}
                  className={`calendar-day ${dayClass} ${hasItems ? "calendar-day-has-items" : "calendar-day-no-items"}`}
                >
                  <div className="calendar-day-header">
                    <div className="calendar-day-date-label">{dayLabel}</div>
                    <div className="calendar-day-number">{cell.day}</div>
                    {successRateLabel ? <div className="calendar-day-success-rate">{successRateLabel}</div> : null}
                  </div>
                  <div className="calendar-events">
                    {(cell.items || []).map((item) => (
                      <div key={item.id} className="calendar-event">
                        {item.kind === "upcoming" ? (
                          <>
                            <div className="calendar-event-matchup">
                              <TeamMatchup
                                league={league}
                                awayTeamCode={item.primaryTeam}
                                homeTeamCode={item.secondaryTeam}
                                awayLabel={item.primaryTeam}
                                homeLabel={item.secondaryTeam}
                                separator="vs"
                              />
                            </div>
                            <div className="calendar-event-status">
                              <span className={`status-dot ${item.dotClass}`} />
                              <span>{item.status}</span>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="calendar-event-status">
                              <span className={`status-dot ${item.dotClass}`} />
                              <span>{item.status}</span>
                            </div>
                            <div className="calendar-event-matchup">
                              <TeamMatchup
                                league={league}
                                awayTeamCode={item.primaryTeam}
                                homeTeamCode={item.secondaryTeam}
                                awayLabel={item.primaryTeam}
                                homeLabel={item.secondaryTeam}
                                separator="vs"
                              />
                            </div>
                          </>
                        )}
                        {item.detail ? <div className="calendar-event-detail">{item.detail}</div> : null}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ActualVsExpectedPage() {
  return (
    <Suspense fallback={<p className="small">Loading actual vs expected...</p>}>
      <ActualVsExpectedPageContent />
    </Suspense>
  );
}
