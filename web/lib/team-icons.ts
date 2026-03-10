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

const NBA_TEAM_ICON_CODES = [
  "ATL",
  "BKN",
  "BOS",
  "CHA",
  "CHI",
  "CLE",
  "DAL",
  "DEN",
  "DET",
  "GS",
  "HOU",
  "IND",
  "LAC",
  "LAL",
  "MEM",
  "MIA",
  "MIL",
  "MIN",
  "NO",
  "NY",
  "OKC",
  "ORL",
  "PHI",
  "PHX",
  "POR",
  "SA",
  "SAC",
  "TOR",
  "UTAH",
  "WSH",
] as const;

const NHL_TEAM_ICON_CODES = [
  "ANA",
  "BOS",
  "BUF",
  "CAR",
  "CBJ",
  "CGY",
  "CHI",
  "COL",
  "DAL",
  "DET",
  "EDM",
  "FLA",
  "LAK",
  "MIN",
  "MTL",
  "NJD",
  "NSH",
  "NYI",
  "NYR",
  "OTT",
  "PHI",
  "PIT",
  "SEA",
  "SJS",
  "STL",
  "TBL",
  "TOR",
  "UTA",
  "VAN",
  "VGK",
  "WPG",
  "WSH",
] as const;

function buildTeamIconMap(pathPrefix: string, extension: "png" | "svg", codes: readonly string[]): Record<string, TeamIconDefinition> {
  return Object.fromEntries(
    codes.map((code) => [
      code,
      {
        ...DEFAULT_ICON,
        src: `${pathPrefix}/${code.toLowerCase()}.${extension}`,
      },
    ])
  );
}

const TEAM_ICONS: Record<LeagueCode, Record<string, TeamIconDefinition>> = {
  NBA: buildTeamIconMap("/team-icons/nba", "png", NBA_TEAM_ICON_CODES),
  NHL: buildTeamIconMap("/team-icons/nhl", "svg", NHL_TEAM_ICON_CODES),
  NCAAM: {},
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

export function resolveTeamIconSrc(src?: string): string | undefined {
  if (!src) return undefined;
  if (/^(?:https?:|data:)/.test(src)) return src;

  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
  const normalized = src.startsWith("/") ? src : `/${src}`;
  return `${basePath}${normalized}`;
}

export function getTeamIconDefinition(league: LeagueCode, teamCode?: string | null): TeamIconDefinition {
  const normalized = String(teamCode || "").trim().toUpperCase();
  return TEAM_ICONS[league]?.[normalized] || DEFAULT_ICON;
}
