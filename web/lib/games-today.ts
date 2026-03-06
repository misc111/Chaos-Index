type ScheduledGameDateLike = {
  game_date_utc?: string | null;
  start_time_utc?: string | null;
};

const CENTRAL_TIME_ZONE = "America/Chicago";
const DATE_LABEL_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  month: "short",
  day: "numeric",
  year: "numeric",
});

export function normalizeUtcTimestamp(value: string): string {
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

export function normalizeCentralDateKey(value?: string | null): string | null {
  const trimmed = String(value || "").trim();
  return /^\d{4}-\d{2}-\d{2}$/.test(trimmed) ? trimmed : null;
}

function dateKeyFromDate(date: Date, timeZone: string): string | null {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);

  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) return null;
  return `${year}-${month}-${day}`;
}

function parseCentralDateKey(dateKey: string): Date | null {
  const normalized = normalizeCentralDateKey(dateKey);
  if (!normalized) return null;

  const [year, month, day] = normalized.split("-").map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day, 12));
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function centralDateKeyFromTimestamp(value?: string | null): string | null {
  if (!value) return null;
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return null;
  return dateKeyFromDate(parsed, CENTRAL_TIME_ZONE);
}

export function centralTodayDateKey(): string {
  return dateKeyFromDate(new Date(), CENTRAL_TIME_ZONE) || new Date().toISOString().slice(0, 10);
}

export function dateKeyForScheduledGame(row: ScheduledGameDateLike): string | null {
  const byStartTime = centralDateKeyFromTimestamp(row.start_time_utc);
  if (byStartTime) return byStartTime;

  const fallback = String(row.game_date_utc || "").trim();
  if (!fallback) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(fallback)) return fallback;

  const byGameDateTimestamp = centralDateKeyFromTimestamp(fallback);
  if (byGameDateTimestamp) return byGameDateTimestamp;

  return fallback.length >= 10 ? fallback.slice(0, 10) : null;
}

export function shiftCentralDateKey(dateKey: string, dayOffset: number): string {
  const parsed = parseCentralDateKey(dateKey);
  if (!parsed) return dateKey;
  parsed.setUTCDate(parsed.getUTCDate() + dayOffset);
  return dateKeyFromDate(parsed, "UTC") || dateKey;
}

export function formatCentralDateLabel(dateKey: string): string {
  const parsed = parseCentralDateKey(dateKey);
  if (!parsed) return dateKey;
  return DATE_LABEL_FORMATTER.format(parsed);
}

export function formatCentralDateSummary(dateKey: string, todayKey = centralTodayDateKey()): string {
  if (dateKey === todayKey) {
    return "today";
  }
  return formatCentralDateLabel(dateKey);
}
