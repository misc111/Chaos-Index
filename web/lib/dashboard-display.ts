import { normalizeUtcTimestamp } from "@/lib/games-today";

export function formatCentralTimestamp(value?: string | null): string {
  if (!value) return "Unavailable";
  const parsed = new Date(normalizeUtcTimestamp(value));
  if (Number.isNaN(parsed.getTime())) return value;
  return `${parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "America/Chicago",
  })} ${parsed.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Chicago",
  })} CT`;
}

export function formatCentralDate(value?: string | null): string {
  if (!value) return "selected date";
  const parsed = new Date(`${value}T12:00:00-05:00`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "America/Chicago",
  });
}

export function shortenToken(value?: string | null, visible = 18): string {
  const text = String(value || "").trim();
  if (!text) return "-";
  if (text.length <= visible) return text;
  return `${text.slice(0, visible)}...`;
}

export function describeFreshForecastStatus(args: {
  status?: string | null;
  scheduledGameCount?: number | null;
  dateCentral?: string | null;
}): { headline: string; detail: string } {
  const dateLabel = formatCentralDate(args.dateCentral);
  const scheduled = Number(args.scheduledGameCount || 0);

  if (args.status === "no_slate") {
    return {
      headline: `No MLB slate for ${dateLabel}`,
      detail: "There are no scheduled games in this snapshot, so betting rows are intentionally empty.",
    };
  }

  if (args.status === "missing") {
    return {
      headline: "Forecast snapshot missing",
      detail: `${scheduled || "Scheduled"} MLB game${scheduled === 1 ? "" : "s"} found, but no fresh forecast snapshot is available yet.`,
    };
  }

  if (args.status === "available") {
    return {
      headline: "Forecast snapshot available",
      detail: "Pregame forecasts are available for the selected slate.",
    };
  }

  return {
    headline: "Snapshot status pending",
    detail: "The dashboard has not received enough snapshot metadata to explain this state yet.",
  };
}
