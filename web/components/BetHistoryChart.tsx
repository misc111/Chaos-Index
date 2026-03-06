"use client";

import type { HistoricalDailyPoint } from "@/lib/bet-history-types";
import { formatUsd } from "@/lib/currency";
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

  return (
    <div className={`card ${styles.chartCard}`}>
      <div>
        <h2 className="title">Cumulative Bet Replay</h2>
        <p className="small">Daily running P/L using the same bet rules as the live Games Today table.</p>
      </div>

      <div className={styles.chartWrap}>
        <svg viewBox={`0 0 ${width} ${height}`} className={styles.chartSvg} role="img" aria-label="Cumulative bet replay line chart">
          <defs>
            <linearGradient id="bet-history-line" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0f766e" />
              <stop offset="100%" stopColor="#2563eb" />
            </linearGradient>
          </defs>

          <rect x="0" y="0" width={width} height={height} rx="14" fill="transparent" />

          {yTicks.map((tick) => {
            const y = padTop + (1 - (tick - minY) / span) * plotHeight;
            return (
              <g key={tick}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                <text x={padLeft - 10} y={y + 4} textAnchor="end" fill="#64748b" fontSize="11">
                  {formatUsd(tick, { minimumFractionDigits: 2 })}
                </text>
              </g>
            );
          })}

          <line x1={padLeft} y1={zeroY} x2={width - padRight} y2={zeroY} stroke="#94a3b8" strokeDasharray="5 4" />

          <path d={linePath} fill="none" stroke="url(#bet-history-line)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />

          {coords.map((coord) => (
            <circle
              key={coord.point.date_central}
              cx={coord.x}
              cy={coord.y}
              r="4"
              fill={coord.point.cumulative_profit >= 0 ? "#0f766e" : "#b91c1c"}
              stroke="#ffffff"
              strokeWidth="2"
            />
          ))}

          {xTickIndexes.map((index) => {
            const coord = coords[index];
            return (
              <text key={coord.point.date_central} x={coord.x} y={height - 12} textAnchor="middle" fill="#64748b" fontSize="11">
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
