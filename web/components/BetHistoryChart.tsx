"use client";

import { useId, useState } from "react";

import { BET_UNIT_DOLLARS, REFERENCE_BANKROLL_DOLLARS } from "@/lib/betting";
import type { HistoricalDailyPoint } from "@/lib/bet-history-types";
import { formatSignedUsd, formatUsd } from "@/lib/currency";
import styles from "./BetHistory.module.css";

type Props = {
  points: HistoricalDailyPoint[];
};

function formatDateShort(dateKey: string): string {
  const parsed = new Date(`${dateKey}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return dateKey;
  return parsed.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export default function BetHistoryChart({ points }: Props) {
  const [activePointIndex, setActivePointIndex] = useState<number | null>(null);
  const chartId = useId().replace(/:/g, "");

  if (!points.length) {
    return (
      <div className={`card ${styles.chartCard}`}>
        <h2 className="title">Cumulative Bet Replay</h2>
        <p className={styles.emptyState}>No settled simulated bets are available yet.</p>
      </div>
    );
  }

  const width = 960;
  const height = 280;
  const padLeft = 58;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 38;
  const plottedValues = points.map((point) => point.cumulative_profit);
  const minY = Math.min(0, ...plottedValues);
  const maxY = Math.max(0, ...plottedValues);
  const span = Math.max(maxY - minY, 1);
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const coords = points.map((point, index) => {
    const x = padLeft + (index / Math.max(points.length - 1, 1)) * plotWidth;
    const y = padTop + (1 - (point.cumulative_profit - minY) / span) * plotHeight;
    return { x, y, point };
  });

  const linePath = coords.map((coord, index) => `${index === 0 ? "M" : "L"}${coord.x},${coord.y}`).join(" ");
  const zeroY = padTop + (1 - (0 - minY) / span) * plotHeight;
  const yTicks = Array.from({ length: 5 }, (_, index) => minY + (span * index) / 4);
  const xTickIndexes = Array.from(new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])).filter(
    (index) => index >= 0 && index < points.length
  );
  const lastPoint = points[points.length - 1];
  const activeCoord = activePointIndex === null ? null : coords[activePointIndex];
  const tooltipBelow = activeCoord ? activeCoord.y < padTop + 34 : false;
  const gradientId = `bet-history-line-${chartId}`;

  return (
    <div className={`card ${styles.chartCard}`}>
      <div>
        <h2 className="title">Cumulative Bet Replay</h2>
        <p className="small">
          Daily running P/L using the same rules as the live Games Today table. One unit is {formatUsd(BET_UNIT_DOLLARS)} on a {formatUsd(REFERENCE_BANKROLL_DOLLARS)} reference bankroll.
        </p>
      </div>

      <div className={styles.chartWrap}>
        {activeCoord ? (
          <div
            className={`${styles.chartTooltip} ${tooltipBelow ? styles.chartTooltipBelow : ""}`}
            style={{
              left: `${(activeCoord.x / width) * 100}%`,
              top: `${(activeCoord.y / height) * 100}%`,
            }}
          >
            <p className={styles.chartTooltipDate}>{formatDateShort(activeCoord.point.date_central)}</p>
            <p className={styles.chartTooltipValue}>
              Net {formatSignedUsd(activeCoord.point.cumulative_profit, { minimumFractionDigits: 2 })}
            </p>
            <p className={styles.chartTooltipDetail}>
              Day {formatSignedUsd(activeCoord.point.daily_profit, { minimumFractionDigits: 2 })}
            </p>
          </div>
        ) : null}

        <svg
          viewBox={`0 0 ${width} ${height}`}
          className={styles.chartSvg}
          role="img"
          aria-label="Cumulative bet replay line chart"
          onPointerLeave={() => setActivePointIndex(null)}
        >
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="var(--chart-line-start)" />
              <stop offset="100%" stopColor="var(--chart-line-end)" />
            </linearGradient>
          </defs>

          <rect x="0" y="0" width={width} height={height} rx="14" fill="transparent" />

          {yTicks.map((tick) => {
            const y = padTop + (1 - (tick - minY) / span) * plotHeight;
            return (
              <g key={tick}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="var(--chart-grid)" strokeWidth="1" />
                <text x={padLeft - 10} y={y + 4} textAnchor="end" fill="var(--chart-axis)" fontSize="11">
                  {formatUsd(tick, { minimumFractionDigits: 2 })}
                </text>
              </g>
            );
          })}

          <line x1={padLeft} y1={zeroY} x2={width - padRight} y2={zeroY} stroke="var(--chart-baseline)" strokeDasharray="5 4" />

          <path d={linePath} fill="none" stroke={`url(#${gradientId})`} strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />

          {coords.map((coord, index) => (
            <g key={coord.point.date_central}>
              <circle
                cx={coord.x}
                cy={coord.y}
                r="14"
                fill="transparent"
                className={styles.chartPointHit}
                tabIndex={0}
                aria-label={`${formatDateShort(coord.point.date_central)} cumulative net ${formatSignedUsd(coord.point.cumulative_profit, {
                  minimumFractionDigits: 2,
                })}`}
                onPointerEnter={() => setActivePointIndex(index)}
                onPointerDown={() => setActivePointIndex(index)}
                onFocus={() => setActivePointIndex(index)}
                onBlur={() => setActivePointIndex((currentIndex) => (currentIndex === index ? null : currentIndex))}
              />
              <circle
                cx={coord.x}
                cy={coord.y}
                r={activePointIndex === index ? 5.5 : 4}
                fill={coord.point.cumulative_profit >= 0 ? "var(--chart-point-positive)" : "var(--chart-point-negative)"}
                stroke="var(--chart-point-stroke)"
                strokeWidth={activePointIndex === index ? "2.5" : "2"}
                className={activePointIndex === index ? styles.chartPointActive : undefined}
              />
            </g>
          ))}

          {xTickIndexes.map((index) => {
            const coord = coords[index];
            return (
              <text key={coord.point.date_central} x={coord.x} y={height - 12} textAnchor="middle" fill="var(--chart-axis)" fontSize="11">
                {formatDateShort(coord.point.date_central)}
              </text>
            );
          })}
        </svg>
      </div>

      <div className={styles.chartMeta}>
        <span>
          Latest net:{" "}
          <span className={styles.chartMetaStrong}>{formatUsd(lastPoint.cumulative_profit, { minimumFractionDigits: 2 })}</span>
        </span>
        <span>
          Risked:{" "}
          <span className={styles.chartMetaStrong}>
            {formatUsd(points.reduce((sum, point) => sum + point.risked, 0), { minimumFractionDigits: 2 })}
          </span>
        </span>
        <span>
          Tracked days: <span className={styles.chartMetaStrong}>{points.length}</span>
        </span>
      </div>
    </div>
  );
}
