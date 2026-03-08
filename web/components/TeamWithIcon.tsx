import Image from "next/image";
import type { CSSProperties } from "react";
import type { LeagueCode } from "@/lib/league";
import { getTeamIconDefinition, normalizeTeamCode, resolveTeamIconSrc } from "@/lib/team-icons";
import styles from "./TeamWithIcon.module.css";

type TeamWithIconProps = {
  league: LeagueCode;
  teamCode?: string | null;
  label?: string | null;
  size?: "sm" | "md";
  className?: string;
  textClassName?: string;
};

type TeamMatchupProps = {
  league: LeagueCode;
  awayTeamCode?: string | null;
  homeTeamCode?: string | null;
  awayLabel?: string | null;
  homeLabel?: string | null;
  separator?: string;
  size?: "sm" | "md";
  className?: string;
};

type BetStakeWithIconProps = {
  league: LeagueCode;
  teamCode?: string | null;
  label?: string | null;
  stake: number;
  size?: "sm" | "md";
  zeroLabel?: string;
  className?: string;
};

function joinClassNames(...values: Array<string | undefined>): string {
  return values.filter(Boolean).join(" ");
}

function fallbackLetters(teamCode: string, label: string): string {
  const compactCode = teamCode.replace(/[^A-Z0-9]/g, "");
  if (compactCode) return compactCode.slice(0, 3);
  const compactLabel = label.replace(/[^A-Z0-9]/g, "");
  return compactLabel.slice(0, 3) || "TM";
}

function formatStakeAmount(stake: number): string {
  if (!Number.isFinite(stake) || stake <= 0) return "$0";
  const fractionDigits = Number.isInteger(stake) ? 0 : 2;
  return `$${stake.toFixed(fractionDigits)}`;
}

export default function TeamWithIcon({
  league,
  teamCode,
  label,
  size = "sm",
  className,
  textClassName,
}: TeamWithIconProps) {
  const displayLabel = String(label || teamCode || "").trim() || "Unknown team";
  const normalizedCode = normalizeTeamCode(teamCode, displayLabel);
  const icon = getTeamIconDefinition(league, normalizedCode);
  const iconSize = size === "md" ? 24 : 20;
  const iconSrc = resolveTeamIconSrc(icon.src);
  const frameStyle = {
    "--team-icon-background": icon.background,
    "--team-icon-border": icon.border,
    "--team-icon-text": icon.text,
  } as CSSProperties;

  return (
    <span className={joinClassNames(styles.teamWithIcon, size === "md" ? styles.sizeMd : styles.sizeSm, className)}>
      <span
        className={joinClassNames(styles.iconFrame, iconSrc ? styles.logoBox : styles.fallbackFrame)}
        style={iconSrc ? undefined : frameStyle}
        aria-hidden="true"
      >
        {iconSrc ? (
          <Image
            src={iconSrc}
            alt=""
            width={iconSize}
            height={iconSize}
            className={styles.iconImage}
            unoptimized
          />
        ) : (
          <span className={styles.fallbackText}>{fallbackLetters(normalizedCode, displayLabel.toUpperCase())}</span>
        )}
      </span>
      <span className={joinClassNames(styles.teamText, textClassName)}>{displayLabel}</span>
    </span>
  );
}

export function TeamMatchup({
  league,
  awayTeamCode,
  homeTeamCode,
  awayLabel,
  homeLabel,
  separator = "at",
  size = "sm",
  className,
}: TeamMatchupProps) {
  return (
    <span className={joinClassNames(styles.matchup, className)}>
      <TeamWithIcon league={league} teamCode={awayTeamCode} label={awayLabel} size={size} />
      <span className={styles.separator}>{separator}</span>
      <TeamWithIcon league={league} teamCode={homeTeamCode} label={homeLabel} size={size} />
    </span>
  );
}

export function BetStakeWithIcon({
  league,
  teamCode,
  label,
  stake,
  size = "sm",
  zeroLabel = "$0",
  className,
}: BetStakeWithIconProps) {
  const displayLabel = String(label || teamCode || "").trim();
  if (!displayLabel || !Number.isFinite(stake) || stake <= 0) {
    return <span className={className}>{zeroLabel}</span>;
  }

  return (
    <span className={joinClassNames(styles.betStake, className)}>
      <span className={styles.stakeAmount}>{formatStakeAmount(stake)}</span>
      <TeamWithIcon league={league} teamCode={teamCode} label={displayLabel} size={size} />
    </span>
  );
}
