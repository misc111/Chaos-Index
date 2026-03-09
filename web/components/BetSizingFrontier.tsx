"use client";

import type { KeyboardEvent } from "react";
import type { BetSizingPolicyPreview } from "@/lib/bet-sizing-view";
import styles from "./BetSizingFrontier.module.css";

type Props = {
  points: BetSizingPolicyPreview[];
  selectedKey: string;
  officialPolicy?: BetSizingPolicyPreview | null;
  onSelect: (key: string) => void;
};

function formatUnits(value: number): string {
  return value.toFixed(2);
}

function formatScore(value: number): string {
  return value.toFixed(2);
}

function handleKey(event: KeyboardEvent<SVGGElement>, key: string, onSelect: (key: string) => void) {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }

  event.preventDefault();
  onSelect(key);
}

function scoreColor(value: number, min: number, max: number): string {
  const span = Math.max(max - min, 1e-6);
  const ratio = Math.min(1, Math.max(0, (value - min) / span));
  const hue = 208 - ratio * 156;
  const lightness = 52 + ratio * 8;
  return `hsl(${hue} 78% ${lightness}%)`;
}

export default function BetSizingFrontier({ points, selectedKey, officialPolicy, onSelect }: Props) {
  const plottedPoints = points.filter((point) => point.metrics);
  if (!plottedPoints.length) {
    return (
      <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <div>
          <h2 className="title">Replay Policy Map</h2>
          <p className="small">Replay-tested policy comparisons appear here once enough matched replay data is available.</p>
        </div>
      </div>
      </div>
    );
  }

  const width = 920;
  const height = 360;
  const padLeft = 68;
  const padRight = 30;
  const padTop = 24;
  const padBottom = 56;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const volValues = plottedPoints.map((point) => point.metrics?.daily_volatility_units ?? 0);
  const returnValues = plottedPoints.map((point) => point.metrics?.mean_daily_profit_units ?? 0);
  const sharpeValues = plottedPoints.map((point) => point.metrics?.sharpe_ratio ?? 0);
  const offFrontierMetrics =
    officialPolicy && !officialPolicy.isFrontierPoint && officialPolicy.metrics ? officialPolicy.metrics : null;

  if (offFrontierMetrics) {
    volValues.push(offFrontierMetrics.daily_volatility_units);
    returnValues.push(offFrontierMetrics.mean_daily_profit_units);
    sharpeValues.push(offFrontierMetrics.sharpe_ratio);
  }

  const minX = Math.min(...volValues);
  const maxX = Math.max(...volValues);
  const minY = Math.min(0, ...returnValues);
  const maxY = Math.max(...returnValues);
  const minSharpe = Math.min(...sharpeValues);
  const maxSharpe = Math.max(...sharpeValues);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);

  const coords = plottedPoints.map((point) => {
    const metrics = point.metrics!;
    const x = padLeft + ((metrics.daily_volatility_units - minX) / spanX) * plotWidth;
    const y = padTop + (1 - (metrics.mean_daily_profit_units - minY) / spanY) * plotHeight;
    return { point, x, y };
  });

  const linePath = coords.map((coord, index) => `${index === 0 ? "M" : "L"}${coord.x},${coord.y}`).join(" ");
  const xTicks = Array.from({ length: 5 }, (_, index) => minX + (spanX * index) / 4);
  const yTicks = Array.from({ length: 5 }, (_, index) => minY + (spanY * index) / 4);

  const offFrontierCoord = offFrontierMetrics
    ? {
        x: padLeft + ((offFrontierMetrics.daily_volatility_units - minX) / spanX) * plotWidth,
        y: padTop + (1 - (offFrontierMetrics.mean_daily_profit_units - minY) / spanY) * plotHeight,
      }
    : null;

  return (
    <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Stage 1</p>
          <h2 className="title">Choose a Risk Style</h2>
          <p className="small">
            Each dot is one replay-tested policy. Farther right means more swingy. Higher means better average daily profit.
          </p>
        </div>
        <div className={styles.legend}>
          <span className={styles.legendSwatchLow} />
          <span className="small">Lower risk-adjusted score</span>
          <span className={styles.legendSwatchHigh} />
          <span className="small">Higher risk-adjusted score</span>
        </div>
      </div>

      <div className={styles.chartWrap}>
        <svg viewBox={`0 0 ${width} ${height}`} className={styles.chart} role="img" aria-label="Historical replay policy map">
          {yTicks.map((tick) => {
            const y = padTop + (1 - (tick - minY) / spanY) * plotHeight;
            return (
              <g key={`y-${tick.toFixed(4)}`}>
                <line x1={padLeft} y1={y} x2={width - padRight} y2={y} className={styles.gridLine} />
                <text x={padLeft - 10} y={y + 4} textAnchor="end" className={styles.axisText}>
                  {formatUnits(tick)}
                </text>
              </g>
            );
          })}

          {xTicks.map((tick) => {
            const x = padLeft + ((tick - minX) / spanX) * plotWidth;
            return (
              <g key={`x-${tick.toFixed(4)}`}>
                <line x1={x} y1={padTop} x2={x} y2={height - padBottom} className={styles.gridLine} />
                <text x={x} y={height - padBottom + 20} textAnchor="middle" className={styles.axisText}>
                  {formatUnits(tick)}
                </text>
              </g>
            );
          })}

          <path d={linePath} className={styles.frontierLine} />

          {coords.map(({ point, x, y }) => {
            const metrics = point.metrics!;
            const isSelected = point.configSignature === selectedKey;
            const color = scoreColor(metrics.sharpe_ratio, minSharpe, maxSharpe);

            return (
              <g
                key={point.configSignature}
                role="button"
                tabIndex={0}
                className={styles.pointButton}
                onClick={() => onSelect(point.configSignature)}
                onKeyDown={(event) => handleKey(event, point.configSignature, onSelect)}
                aria-label={`${point.label}. Volatility ${formatUnits(metrics.daily_volatility_units)}, return ${formatUnits(metrics.mean_daily_profit_units)}, risk-adjusted score ${formatScore(metrics.sharpe_ratio)}.`}
              >
                <circle cx={x} cy={y} r={isSelected ? 14 : 10} className={styles.pointHalo} />
                <circle cx={x} cy={y} r={isSelected ? 8.5 : 6.5} fill={color} className={styles.pointCore} />
                {point.matchingStrategies.length ? (
                  <text x={x} y={y - 16} textAnchor="middle" className={styles.pointLabel}>
                    {point.matchingStrategies.join(" / ")}
                  </text>
                ) : null}
                <title>{`${point.label}: volatility ${formatUnits(metrics.daily_volatility_units)}, return ${formatUnits(metrics.mean_daily_profit_units)}, risk-adjusted score ${formatScore(metrics.sharpe_ratio)}`}</title>
              </g>
            );
          })}

          {offFrontierCoord && officialPolicy?.metrics ? (
            <g
              role="button"
              tabIndex={0}
              className={styles.pointButton}
              onClick={() => onSelect(officialPolicy.configSignature)}
              onKeyDown={(event) => handleKey(event, officialPolicy.configSignature, onSelect)}
              aria-label={`${officialPolicy.label}. Off-frontier profile with volatility ${formatUnits(officialPolicy.metrics.daily_volatility_units)}, return ${formatUnits(officialPolicy.metrics.mean_daily_profit_units)}.`}
            >
              <path
                d={`M ${offFrontierCoord.x} ${offFrontierCoord.y - 10} L ${offFrontierCoord.x + 10} ${offFrontierCoord.y} L ${offFrontierCoord.x} ${offFrontierCoord.y + 10} L ${offFrontierCoord.x - 10} ${offFrontierCoord.y} Z`}
                className={officialPolicy.configSignature === selectedKey ? styles.offFrontierSelected : styles.offFrontier}
              />
              <text x={offFrontierCoord.x} y={offFrontierCoord.y - 18} textAnchor="middle" className={styles.pointLabel}>
                Saved profile
              </text>
              <title>{`${officialPolicy.label}: selected outside the main replay map using downside criteria.`}</title>
            </g>
          ) : null}

          <text x={padLeft + plotWidth / 2} y={height - 12} textAnchor="middle" className={styles.axisTitle}>
            Daily volatility (bet units)
          </text>
          <text
            x={18}
            y={padTop + plotHeight / 2}
            transform={`rotate(-90 18 ${padTop + plotHeight / 2})`}
            textAnchor="middle"
            className={styles.axisTitle}
          >
            Mean daily profit (bet units)
          </text>
        </svg>
      </div>

      <p className="small">
        This replay map is a heuristic ranking built from matched historical replay using continuous sizing. Click a dot to preview how a different risk point would size the slate.
      </p>
    </div>
  );
}
