export type LeagueCode = "NHL" | "NBA" | "NCAAM";

export const ALL_LEAGUES: LeagueCode[] = ["NBA", "NHL", "NCAAM"];

export function normalizeLeague(value?: string | null): LeagueCode {
  const token = String(value || "").trim().toUpperCase();
  if (token === "NCAAM" || token === "NCAA") return "NCAAM";
  if (token === "NHL") return "NHL";
  return "NBA";
}

export function leagueFromRequest(request: Request): LeagueCode {
  const url = new URL(request.url);
  return normalizeLeague(url.searchParams.get("league"));
}

export function withLeague(path: string, league: LeagueCode): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}league=${league}`;
}

export function displayLeagueLabel(league: LeagueCode): string {
  return league === "NCAAM" ? "NCAA" : league;
}
