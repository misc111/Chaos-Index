export const TEAM_ICON_SIZE_TOKENS = {
  sm: {
    gapRem: 0.4,
    imagePx: 24,
    logoBoxRem: 1.4,
    fallbackBoxRem: 1.2,
    fallbackFontRem: 0.54,
  },
  md: {
    gapRem: 0.45,
    imagePx: 28,
    logoBoxRem: 1.7,
    fallbackBoxRem: 1.45,
    fallbackFontRem: 0.62,
  },
} as const;

export type TeamIconSize = keyof typeof TEAM_ICON_SIZE_TOKENS;
