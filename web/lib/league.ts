export type LeagueCode = "NHL" | "NBA";

export function normalizeLeague(value?: string | null): LeagueCode {
  return String(value || "").trim().toUpperCase() === "NBA" ? "NBA" : "NHL";
}

export function leagueFromRequest(request: Request): LeagueCode {
  const url = new URL(request.url);
  return normalizeLeague(url.searchParams.get("league"));
}

export function withLeague(path: string, league: LeagueCode): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}league=${league}`;
}
