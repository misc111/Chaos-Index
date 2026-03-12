import { ALL_LEAGUES, LEAGUE_ALIASES, LEAGUE_RUNTIME, type LeagueCode } from "@/lib/generated/league-registry";

export { ALL_LEAGUES, type LeagueCode };

export function normalizeLeague(value?: string | null): LeagueCode {
  const token = String(value || "").trim().toUpperCase();
  return LEAGUE_ALIASES[token] || "NBA";
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
  return LEAGUE_RUNTIME[league].displayLabel;
}
