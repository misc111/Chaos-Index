"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

import {
  HISTORICAL_BANKROLL_START_DATE_CENTRAL,
  HISTORICAL_BANKROLL_START_DOLLARS,
} from "@/lib/betting";
import type { HistoricalDailyPoint } from "@/lib/bet-history-types";
import { formatSignedUsd, formatUsd } from "@/lib/currency";
import styles from "./BetHistory.module.css";

type Props = {
  points: HistoricalDailyPoint[];
};

type ChartCoord = {
  x: number;
  y: number;
  point: HistoricalDailyPoint;
};

type ChartGeometry = {
  coords: ChartCoord[];
  startingBankrollY: number;
  minY: number;
  span: number;
  plotHeight: number;
  yTicks: number[];
  xTickIndexes: number[];
};

const CHART_WIDTH = 960;
const CHART_HEIGHT = 280;
const CHART_PAD_LEFT = 58;
const CHART_PAD_RIGHT = 20;
const CHART_PAD_TOP = 20;
const CHART_PAD_BOTTOM = 38;
const CHART_ANIMATION_DURATION_MS = 320;
const EMPTY_CHART_GEOMETRY: ChartGeometry = {
  coords: [],
  startingBankrollY: CHART_PAD_TOP,
  minY: HISTORICAL_BANKROLL_START_DOLLARS,
  span: 1,
  plotHeight: CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM,
  yTicks: [],
  xTickIndexes: [],
};

function formatDateShort(dateKey: string): string {
  const parsed = new Date(`${dateKey}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return dateKey;
  return parsed.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function buildChartGeometry(points: HistoricalDailyPoint[]): ChartGeometry {
  const plottedValues = points.map((point) => point.cumulative_bankroll);
  const minY = Math.min(HISTORICAL_BANKROLL_START_DOLLARS, ...plottedValues);
  const maxY = Math.max(HISTORICAL_BANKROLL_START_DOLLARS, ...plottedValues);
  const span = Math.max(maxY - minY, 1);
  const plotWidth = CHART_WIDTH - CHART_PAD_LEFT - CHART_PAD_RIGHT;
  const plotHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;

  const coords = points.map((point, index) => {
    const x = CHART_PAD_LEFT + (index / Math.max(points.length - 1, 1)) * plotWidth;
    const y = CHART_PAD_TOP + (1 - (point.cumulative_bankroll - minY) / span) * plotHeight;
    return { x, y, point };
  });

  const startingBankrollY =
    CHART_PAD_TOP + (1 - (HISTORICAL_BANKROLL_START_DOLLARS - minY) / span) * plotHeight;
  const yTicks = Array.from({ length: 5 }, (_, index) => minY + (span * index) / 4);
  const xTickIndexes = Array.from(new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])).filter(
    (index) => index >= 0 && index < points.length
  );

  return {
    coords,
    startingBankrollY,
    minY,
    span,
    plotHeight,
    yTicks,
    xTickIndexes,
  };
}

function buildLinePath(coords: ChartCoord[]): string {
  return coords.map((coord, index) => `${index === 0 ? "M" : "L"}${coord.x},${coord.y}`).join(" ");
}

function interpolateNumber(from: number, to: number, progress: number): number {
  return from + (to - from) * progress;
}

function easeOutCubic(progress: number): number {
  return 1 - Math.pow(1 - progress, 3);
}

function canAnimateTransition(previousCoords: ChartCoord[], nextCoords: ChartCoord[]): boolean {
  if (!previousCoords.length || !nextCoords.length) return false;
  if (previousCoords.length !== nextCoords.length) return false;
  return previousCoords.every((coord, index) => coord.point.date_central === nextCoords[index]?.point.date_central);
}

export default function BetHistoryChart({ points }: Props) {
  const [activePointIndex, setActivePointIndex] = useState<number | null>(null);
  const chartId = useId().replace(/:/g, "");

  const geometry = useMemo(() => (points.length ? buildChartGeometry(points) : EMPTY_CHART_GEOMETRY), [points]);
  const [displayedCoords, setDisplayedCoords] = useState<ChartCoord[]>(geometry.coords);
  const [displayedStartingBankrollY, setDisplayedStartingBankrollY] = useState<number>(geometry.startingBankrollY);
  const displayedCoordsRef = useRef<ChartCoord[]>(geometry.coords);
  const displayedStartingBankrollYRef = useRef<number>(geometry.startingBankrollY);
  const animationFrameRef = useRef<number | null>(null);
  const reducedMotionRef = useRef<boolean>(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    reducedMotionRef.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const previousCoords = displayedCoordsRef.current;

    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (reducedMotionRef.current || !canAnimateTransition(previousCoords, geometry.coords)) {
      displayedCoordsRef.current = geometry.coords;
      displayedStartingBankrollYRef.current = geometry.startingBankrollY;
      animationFrameRef.current = window.requestAnimationFrame(() => {
        setDisplayedCoords(geometry.coords);
        setDisplayedStartingBankrollY(geometry.startingBankrollY);
        animationFrameRef.current = null;
      });
      return;
    }

    const startBaselineY = displayedStartingBankrollYRef.current;
    let startTime: number | null = null;

    displayedCoordsRef.current = previousCoords;
    displayedStartingBankrollYRef.current = startBaselineY;

    const animate = (timestamp: number) => {
      startTime ??= timestamp;
      const rawProgress = Math.min((timestamp - startTime) / CHART_ANIMATION_DURATION_MS, 1);
      const progress = easeOutCubic(rawProgress);
      const nextCoords = geometry.coords.map((coord, index) => {
        const startCoord = previousCoords[index] ?? coord;
        return {
          x: interpolateNumber(startCoord.x, coord.x, progress),
          y: interpolateNumber(startCoord.y, coord.y, progress),
          point: coord.point,
        };
      });
      const nextBaselineY = interpolateNumber(startBaselineY, geometry.startingBankrollY, progress);

      displayedCoordsRef.current = nextCoords;
      displayedStartingBankrollYRef.current = nextBaselineY;
      setDisplayedCoords(nextCoords);
      setDisplayedStartingBankrollY(nextBaselineY);

      if (rawProgress < 1) {
        animationFrameRef.current = window.requestAnimationFrame(animate);
        return;
      }

      animationFrameRef.current = null;
    };

    animationFrameRef.current = window.requestAnimationFrame(animate);
  }, [geometry]);

  if (!points.length) {
    return (
      <div className={`card ${styles.chartCard}`}>
        <h2 className="title">Cumulative Bankroll</h2>
        <p className={styles.emptyState}>No settled simulated bets are available yet.</p>
      </div>
    );
  }

  const coords = displayedCoords;
  const linePath = buildLinePath(coords);
  const startingBankrollY = displayedStartingBankrollY;
  const { minY, plotHeight, span, xTickIndexes, yTicks } = geometry;
  const displayedXTickIndexes = xTickIndexes.filter((index) => index < coords.length);
  const lastPoint = points[points.length - 1];
  const safeActivePointIndex = activePointIndex !== null && activePointIndex < coords.length ? activePointIndex : null;
  const activeCoord = safeActivePointIndex === null ? null : coords[safeActivePointIndex];
  const tooltipBelow = activeCoord ? activeCoord.y < CHART_PAD_TOP + 34 : false;
  const gradientId = `bet-history-line-${chartId}`;

  return (
    <div className={`card ${styles.chartCard}`}>
      <div>
        <h2 className="title">Cumulative Bankroll</h2>
        <p className="small">
          Bankroll starts at {formatUsd(HISTORICAL_BANKROLL_START_DOLLARS)} on {formatDateShort(HISTORICAL_BANKROLL_START_DATE_CENTRAL)}.
          {" "}Net P/L and ROI above still summarize the same replayed bets.
        </p>
      </div>

      <div className={styles.chartWrap}>
        {activeCoord ? (
          <div
            className={`${styles.chartTooltip} ${tooltipBelow ? styles.chartTooltipBelow : ""}`}
            style={{
              left: `${(activeCoord.x / CHART_WIDTH) * 100}%`,
              top: `${(activeCoord.y / CHART_HEIGHT) * 100}%`,
            }}
          >
            <p className={styles.chartTooltipDate}>{formatDateShort(activeCoord.point.date_central)}</p>
            <p className={styles.chartTooltipValue}>
              Bankroll {formatUsd(activeCoord.point.cumulative_bankroll, { minimumFractionDigits: 2 })}
            </p>
            <p className={styles.chartTooltipDetail}>
              Net {formatSignedUsd(activeCoord.point.cumulative_profit, { minimumFractionDigits: 2 })}
            </p>
            <p className={styles.chartTooltipDetail}>
              Day {formatSignedUsd(activeCoord.point.daily_profit, { minimumFractionDigits: 2 })}
            </p>
            <p className={styles.chartTooltipDetail}>
              Total risked {formatUsd(activeCoord.point.risked, { minimumFractionDigits: 2 })}
            </p>
          </div>
        ) : null}

        <svg
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          className={styles.chartSvg}
          role="img"
          aria-label="Cumulative bankroll line chart"
          onPointerLeave={() => setActivePointIndex(null)}
        >
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="var(--chart-line-start)" />
              <stop offset="100%" stopColor="var(--chart-line-end)" />
            </linearGradient>
          </defs>

          <rect x="0" y="0" width={CHART_WIDTH} height={CHART_HEIGHT} rx="14" fill="transparent" />

          {yTicks.map((tick) => {
            const tickY = CHART_PAD_TOP + (1 - (tick - minY) / span) * plotHeight;
            return (
              <g key={tick}>
                <line
                  x1={CHART_PAD_LEFT}
                  y1={tickY}
                  x2={CHART_WIDTH - CHART_PAD_RIGHT}
                  y2={tickY}
                  stroke="var(--chart-grid)"
                  strokeWidth="1"
                />
                <text x={CHART_PAD_LEFT - 10} y={tickY + 4} textAnchor="end" fill="var(--chart-axis)" fontSize="11">
                  {formatUsd(tick, { minimumFractionDigits: 2 })}
                </text>
              </g>
            );
          })}

          <line
            x1={CHART_PAD_LEFT}
            y1={startingBankrollY}
            x2={CHART_WIDTH - CHART_PAD_RIGHT}
            y2={startingBankrollY}
            stroke="var(--chart-baseline)"
            strokeDasharray="5 4"
          />

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
                aria-label={`${formatDateShort(coord.point.date_central)} bankroll ${formatUsd(coord.point.cumulative_bankroll, {
                  minimumFractionDigits: 2,
                })} cumulative net ${formatSignedUsd(coord.point.cumulative_profit, {
                  minimumFractionDigits: 2,
                })} day result ${formatSignedUsd(coord.point.daily_profit, {
                  minimumFractionDigits: 2,
                })} total risked ${formatUsd(coord.point.risked, { minimumFractionDigits: 2 })}`}
                onPointerEnter={() => setActivePointIndex(index)}
                onPointerDown={() => setActivePointIndex(index)}
                onFocus={() => setActivePointIndex(index)}
                onBlur={() => setActivePointIndex((currentIndex) => (currentIndex === index ? null : currentIndex))}
              />
              <circle
                cx={coord.x}
                cy={coord.y}
                r={safeActivePointIndex === index ? 5.5 : 4}
                fill={coord.point.cumulative_profit >= 0 ? "var(--chart-point-positive)" : "var(--chart-point-negative)"}
                stroke="var(--chart-point-stroke)"
                strokeWidth={safeActivePointIndex === index ? "2.5" : "2"}
                className={safeActivePointIndex === index ? styles.chartPointActive : undefined}
              />
            </g>
          ))}

          {displayedXTickIndexes.map((index) => {
            const coord = coords[index];
            return (
              <text
                key={coord.point.date_central}
                x={coord.x}
                y={CHART_HEIGHT - 12}
                textAnchor="middle"
                fill="var(--chart-axis)"
                fontSize="11"
              >
                {formatDateShort(coord.point.date_central)}
              </text>
            );
          })}
        </svg>
      </div>

      <div className={styles.chartMeta}>
        <span>
          Start bankroll:{" "}
          <span className={styles.chartMetaStrong}>{formatUsd(HISTORICAL_BANKROLL_START_DOLLARS, { minimumFractionDigits: 2 })}</span>
        </span>
        <span>
          Latest bankroll:{" "}
          <span className={styles.chartMetaStrong}>{formatUsd(lastPoint.cumulative_bankroll, { minimumFractionDigits: 2 })}</span>
        </span>
        <span>
          Latest net:{" "}
          <span className={styles.chartMetaStrong}>{formatSignedUsd(lastPoint.cumulative_profit, { minimumFractionDigits: 2 })}</span>
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
