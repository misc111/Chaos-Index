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

const MLB_TEAM_ICON_CODES = [] as const;

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
  MLB: buildTeamIconMap("/team-icons/mlb", "svg", MLB_TEAM_ICON_CODES),
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
