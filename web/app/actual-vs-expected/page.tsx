"use client";

import { useEffect, useMemo, useState } from "react";

type HistoricalRow = {
  game_id: number;
  game_date_utc: string;
  home_team: string;
  away_team: string;
  as_of_utc: string;
  prob_home_win: number;
  predicted_winner: string;
  home_win: number;
  model_correct: number;
};

type UpcomingRow = {
  game_id: number;
  game_date_utc: string;
  home_team: string;
  away_team: string;
  as_of_utc: string;
  ensemble_prob_home_win: number;
  predicted_winner: string;
  start_time_utc?: string | null;
};

type ApiResponse = {
  as_of_utc?: string;
  historical_rows?: HistoricalRow[];
  upcoming_rows?: UpcomingRow[];
};

type CalendarItem = {
  id: string;
  matchup: string;
  status: string;
  detail?: string;
  dotClass: "dot-correct" | "dot-incorrect" | "dot-upcoming";
  kind: "historical" | "upcoming";
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

function formatCentralTime(value?: string | null): string {
  if (!value) return "Time TBD (CT)";
  const normalized = normalizeUtcTimestamp(value);
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return `${value} CT`;
  return parsed.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Chicago",
    timeZoneName: "short",
  });
}

export default function ActualVsExpectedPage() {
  const [historicalRows, setHistoricalRows] = useState<HistoricalRow[]>([]);
  const [upcomingRows, setUpcomingRows] = useState<UpcomingRow[]>([]);
  const [latestAsOf, setLatestAsOf] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  const [monthCursor, setMonthCursor] = useState(() => {
    const now = new Date();
    return { year: now.getFullYear(), month: now.getMonth() };
  });

  const todayKey = useMemo(() => dateKeyFromLocalToday(), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetch("/api/actual-vs-expected", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Request failed: ${res.status}`);
        }
        return res.json() as Promise<ApiResponse>;
      })
      .then((payload) => {
        if (cancelled) return;
        setHistoricalRows(payload.historical_rows || []);
        setUpcomingRows(payload.upcoming_rows || []);
        setLatestAsOf(payload.as_of_utc || "");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const eventsByDate = useMemo(() => {
    const map: Record<string, CalendarItem[]> = {};

    for (const row of historicalRows) {
      const key = normalizeDateKey(row.game_date_utc);
      if (!map[key]) map[key] = [];

      const isCorrect = Number(row.model_correct) === 1;
      map[key].push({
        id: `hist-${row.game_id}`,
        matchup: `${row.home_team} vs ${row.away_team}`,
        status: isCorrect ? "Model correct" : "Model incorrect",
        detail: `Forecast made ${formatAsOfLabel(row.as_of_utc)}`,
        dotClass: isCorrect ? "dot-correct" : "dot-incorrect",
        kind: "historical",
      });
    }

    for (const row of upcomingRows) {
      const key = normalizeDateKey(row.game_date_utc);
      if (!map[key]) map[key] = [];

      const homeWinProb = Number(row.ensemble_prob_home_win);

      map[key].push({
        id: `future-${row.game_id}`,
        matchup: `${row.home_team} vs ${row.away_team}`,
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

              return (
                <div key={cell.key} className={`calendar-day ${dayClass}`}>
                  <div className="calendar-day-number">{cell.day}</div>
                  <div className="calendar-events">
                    {(cell.items || []).map((item) => (
                      <div key={item.id} className="calendar-event">
                        {item.kind === "upcoming" ? (
                          <>
                            <div className="calendar-event-matchup">{item.matchup}</div>
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
                            <div className="calendar-event-matchup">{item.matchup}</div>
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
