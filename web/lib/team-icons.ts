import type { LeagueCode } from "@/lib/league";

type TeamIconDefinition = {
  src?: string;
  background: string;
  border: string;
  text: string;
};

const DEFAULT_ICON: TeamIconDefinition = {
  background: "color-mix(in srgb, var(--surface-field-muted) 88%, white 12%)",
  border: "color-mix(in srgb, var(--border-strong) 76%, white 24%)",
  text: "var(--muted-strong)",
};

const TEAM_ICONS: Record<LeagueCode, Record<string, TeamIconDefinition>> = {
  NBA: {
    TOR: {
      src: "/team-icons/nba/toronto-raptors.svg",
      background: "color-mix(in srgb, #ce1141 18%, white 82%)",
      border: "color-mix(in srgb, #000000 72%, white 28%)",
      text: "#7a0c28",
    },
  },
  NHL: {},
};

export function normalizeTeamCode(teamCode?: string | null, label?: string | null): string {
  const code = String(teamCode || "").trim().toUpperCase();
  if (code) return code;

  const compact = String(label || "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "");
  if (!compact) return "";
  return compact.slice(0, 3);
}

export function getTeamIconDefinition(league: LeagueCode, teamCode?: string | null): TeamIconDefinition {
  const normalized = String(teamCode || "").trim().toUpperCase();
  return TEAM_ICONS[league]?.[normalized] || DEFAULT_ICON;
}
