"use client";

import { useMemo, useState } from "react";

import {
  buildEnsembleSnapshotBankrollSeries,
  listEnsembleSnapshotChartDates,
  type SnapshotBankrollPoint,
  type SnapshotBankrollMode,
  type SnapshotBankrollSeries,
  type SnapshotChartStrategyKey,
} from "@/lib/ensemble-snapshot-chart";
import { HISTORICAL_BANKROLL_START_DOLLARS } from "@/lib/betting";
import { getBetStrategyConfig, type BetStrategy } from "@/lib/betting-strategy";
import { formatSignedUsd, formatUsd } from "@/lib/currency";
import type { EnsembleSnapshotRow } from "@/lib/types";
import chartStyles from "./BetHistory.module.css";
import styles from "./EnsembleSnapshotBankrollChart.module.css";

type Props = {
  snapshots: EnsembleSnapshotRow[];
  activeStrategy?: BetStrategy;
  selectedSnapshotKey: string;
  onSelectSnapshotKey: (snapshotKey: string) => void;
};

type ChartCoord = {
  x: number;
  y: number;
  point: SnapshotBankrollPoint;
};

type ChartSeries = SnapshotBankrollSeries & {
  color: string;
  coords: ChartCoord[];
};

type ChartGeometry = {
  plottedSeries: ChartSeries[];
  startingBankrollY: number;
  minY: number;
  span: number;
  plotHeight: number;
  yTicks: number[];
  xByDate: Map<string, number>;
  xTickDates: string[];
};

const CHART_WIDTH = 960;
const CHART_HEIGHT = 320;
const CHART_PAD_LEFT = 62;
const CHART_PAD_RIGHT = 24;
const CHART_PAD_TOP = 24;
const CHART_PAD_BOTTOM = 42;
const SNAPSHOT_LINE_COLORS = [
  "#0f766e",
  "#1d4ed8",
  "#b45309",
  "#be123c",
  "#6d28d9",
  "#047857",
  "#7c3aed",
  "#c2410c",
  "#0f766e",
  "#374151",
];

function formatDateShort(dateKey: string): string {
  const parsed = new Date(`${dateKey}T12:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return dateKey;
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function pickSeriesColor(index: number): string {
  return SNAPSHOT_LINE_COLORS[index] || `hsl(${(index * 47) % 360} 68% 42%)`;
}

function buildLinePath(coords: ChartCoord[]): string {
  if (!coords.length) return "";
  return coords.map((coord, index) => `${index === 0 ? "M" : "L"}${coord.x},${coord.y}`).join(" ");
}

function buildTickDates(dateKeys: string[]): string[] {
  if (dateKeys.length <= 5) return dateKeys;
  return Array.from(
    new Set([
      dateKeys[0],
      dateKeys[Math.floor((dateKeys.length - 1) * 0.25)],
      dateKeys[Math.floor((dateKeys.length - 1) * 0.5)],
      dateKeys[Math.floor((dateKeys.length - 1) * 0.75)],
      dateKeys[dateKeys.length - 1],
    ])
  );
}

function buildChartGeometry(series: SnapshotBankrollSeries[]): ChartGeometry {
  const dateKeys = listEnsembleSnapshotChartDates(series);
  const plotWidth = CHART_WIDTH - CHART_PAD_LEFT - CHART_PAD_RIGHT;
  const plotHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;
  const xByDate = new Map(
    dateKeys.map((dateKey, index) => [
      dateKey,
      CHART_PAD_LEFT + (index / Math.max(dateKeys.length - 1, 1)) * plotWidth,
    ])
  );
  const plottedValues = series.flatMap((snapshot) => snapshot.points.map((point) => point.cumulative_bankroll));
  const minY = Math.min(HISTORICAL_BANKROLL_START_DOLLARS, ...plottedValues);
  const maxY = Math.max(HISTORICAL_BANKROLL_START_DOLLARS, ...plottedValues);
  const span = Math.max(maxY - minY, 1);
  const startingBankrollY =
    CHART_PAD_TOP + (1 - (HISTORICAL_BANKROLL_START_DOLLARS - minY) / span) * plotHeight;
  const yTicks = Array.from({ length: 5 }, (_, index) => minY + (span * index) / 4);

  // Every snapshot line shares the same date index and bankroll axis. That is
  // what makes the comparison fair: the only thing changing between lines is
  // which frozen model generated the wagers.
  const plottedSeries = series.map((snapshot, index) => ({
    ...snapshot,
    color: pickSeriesColor(index),
    coords: snapshot.points.map((point) => ({
      x: xByDate.get(point.date_central) ?? CHART_PAD_LEFT,
      y: CHART_PAD_TOP + (1 - (point.cumulative_bankroll - minY) / span) * plotHeight,
      point,
    })),
  }));

  return {
    plottedSeries,
    startingBankrollY,
    minY,
    span,
    plotHeight,
    yTicks,
    xByDate,
    xTickDates: buildTickDates(dateKeys),
  };
}

export default function EnsembleSnapshotBankrollChart({
  snapshots,
  activeStrategy = "riskAdjusted",
  selectedSnapshotKey,
  onSelectSnapshotKey,
}: Props) {
  const chartStrategy: SnapshotChartStrategyKey = activeStrategy;
  const [bankrollMode, setBankrollMode] = useState<SnapshotBankrollMode>("independent");
  const activeStrategyConfig = getBetStrategyConfig(chartStrategy);
  const series = useMemo(
    () => buildEnsembleSnapshotBankrollSeries(snapshots, chartStrategy, bankrollMode),
    [snapshots, chartStrategy, bankrollMode]
  );
  const geometry = useMemo(() => buildChartGeometry(series), [series]);
  const selectedSeries =
    geometry.plottedSeries.find((snapshot) => snapshot.snapshot_key === selectedSnapshotKey) || geometry.plottedSeries[0];
  const [hoverState, setHoverState] = useState<{ snapshotKey: string; strategy: SnapshotChartStrategyKey; date: string } | null>(null);

  if (!selectedSeries) {
    return null;
  }

  const activePointDate =
    hoverState &&
    hoverState.snapshotKey === selectedSeries.snapshot_key &&
    hoverState.strategy === chartStrategy &&
    selectedSeries.coords.some((coord) => coord.point.date_central === hoverState.date)
      ? hoverState.date
      : selectedSeries.final_point.date_central;
  const activeCoord =
    selectedSeries.coords.find((coord) => coord.point.date_central === activePointDate) ||
    selectedSeries.coords[selectedSeries.coords.length - 1] ||
    null;
  const tooltipBelow = activeCoord ? activeCoord.y < CHART_PAD_TOP + 40 : false;
  const { minY, plotHeight, span, yTicks, xTickDates, startingBankrollY, xByDate } = geometry;
  const totalSnapshots = geometry.plottedSeries.length;
  const continuityEnabled = bankrollMode === "continuity";

  function selectSnapshot(snapshotKey: string) {
    onSelectSnapshotKey(snapshotKey);
    setHoverState(null);
  }

  return (
    <section className={`card ${chartStyles.chartCard} ${styles.chartCard}`}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Frozen Snapshot Bankroll Paths</p>
          <h2 className="title" style={{ marginTop: 10 }}>
            Every Dated Ensemble Snapshot, Let Loose Through Today
          </h2>
          <p className={styles.body}>
            This mirrors the bet-history bankroll chart, but every line is a different frozen ensemble snapshot. Each one starts from
            the same opening bankroll and then keeps making bets through later slates as if you had never recalibrated again.
          </p>
          <p className={styles.objectivePill}>
            Following Bet Objective: <strong>{activeStrategyConfig.label}</strong>
          </p>
        </div>
        <button
          type="button"
          className={`${styles.continuityButton} ${continuityEnabled ? styles.continuityButtonActive : ""}`}
          aria-pressed={continuityEnabled}
          onClick={() => setBankrollMode((currentMode) => (currentMode === "continuity" ? "independent" : "continuity"))}
        >
          Continuity <span className={styles.continuityState}>{continuityEnabled ? "On" : "Off"}</span>
        </button>
      </div>

      <p className={styles.note}>
        {continuityEnabled
          ? "The y-axis is total bankroll with continuity handoffs turned on. Each new model inherits the prior snapshot's bankroll through D-1, so the lines form one stitched account path across recalibration dates."
          : "The y-axis is total bankroll with isolated replays. Each snapshot restarts at $5,000 so you can compare model quality without carrying forward earlier wins or losses."}{" "}
        This chart follows the dashboard&apos;s Bet Objective buttons, so switching between Balanced, Aggressive, and Conservative redraws
        the full frozen-snapshot bankroll race under that exact rule set.
      </p>

      <div className={chartStyles.chartWrap}>
        {activeCoord ? (
          <div
            className={`${chartStyles.chartTooltip} ${tooltipBelow ? chartStyles.chartTooltipBelow : ""}`}
            style={{
              left: `${(activeCoord.x / CHART_WIDTH) * 100}%`,
              top: `${(activeCoord.y / CHART_HEIGHT) * 100}%`,
            }}
          >
            <p className={chartStyles.chartTooltipDate}>
              Model as of {formatDateShort(selectedSeries.activation_date_central)} · {formatDateShort(activeCoord.point.date_central)}
            </p>
            <p className={chartStyles.chartTooltipValue}>
              Bankroll {formatUsd(activeCoord.point.cumulative_bankroll, { minimumFractionDigits: 2 })}
            </p>
            {activeCoord.point.kind === "daily" ? (
              <>
                <p className={chartStyles.chartTooltipDetail}>
                  {continuityEnabled ? "Account net" : "Net"}{" "}
                  {formatSignedUsd(activeCoord.point.cumulative_profit, { minimumFractionDigits: 2 })}
                </p>
                {continuityEnabled ? (
                  <p className={chartStyles.chartTooltipDetail}>
                    Snapshot-only net{" "}
                    {formatSignedUsd(activeCoord.point.snapshot_cumulative_profit, { minimumFractionDigits: 2 })}
                  </p>
                ) : null}
                <p className={chartStyles.chartTooltipDetail}>
                  Day {formatSignedUsd(activeCoord.point.daily_profit, { minimumFractionDigits: 2 })}
                </p>
                <p className={chartStyles.chartTooltipDetail}>
                  Risked {formatUsd(activeCoord.point.total_risked, { minimumFractionDigits: 2 })} across {activeCoord.point.suggested_bets} bets
                </p>
              </>
            ) : activeCoord.point.kind === "pending" ? (
              <p className={chartStyles.chartTooltipDetail}>No settled games yet for this snapshot.</p>
            ) : (
              <p className={chartStyles.chartTooltipDetail}>
                {continuityEnabled && selectedSeries.starting_bankroll !== HISTORICAL_BANKROLL_START_DOLLARS
                  ? "Opening bankroll after continuity hands off the prior snapshot's end-of-day bankroll."
                  : "Opening bankroll before this frozen model starts settling wagers."}
              </p>
            )}
          </div>
        ) : null}

        <svg
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          className={chartStyles.chartSvg}
          role="img"
          aria-label="Frozen ensemble snapshot bankroll chart"
          onPointerLeave={() => setHoverState(null)}
        >
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

          {geometry.plottedSeries.map((snapshot) => {
            const linePath = buildLinePath(snapshot.coords);
            const isSelected = snapshot.snapshot_key === selectedSeries.snapshot_key;
            return (
              <g key={snapshot.snapshot_key}>
                <path
                  d={linePath}
                  fill="none"
                  stroke={snapshot.color}
                  strokeWidth={isSelected ? "4" : "2.35"}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity={isSelected ? 1 : 0.28}
                />
                <path
                  d={linePath}
                  fill="none"
                  stroke="transparent"
                  strokeWidth="16"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  onPointerEnter={() => selectSnapshot(snapshot.snapshot_key)}
                  onClick={() => selectSnapshot(snapshot.snapshot_key)}
                />
                {snapshot.coords.length ? (
                  <circle
                    cx={snapshot.coords[snapshot.coords.length - 1].x}
                    cy={snapshot.coords[snapshot.coords.length - 1].y}
                    r={isSelected ? 4.8 : 3.2}
                    fill={snapshot.color}
                    stroke="var(--chart-point-stroke)"
                    strokeWidth={isSelected ? "2.4" : "1.8"}
                    opacity={isSelected ? 1 : 0.84}
                  />
                ) : null}
              </g>
            );
          })}

          {selectedSeries.coords.map((coord) => (
            <g key={`${selectedSeries.snapshot_key}-${coord.point.date_central}`}>
              <circle
                cx={coord.x}
                cy={coord.y}
                r="13"
                fill="transparent"
                className={chartStyles.chartPointHit}
                tabIndex={0}
                aria-label={`Model as of ${formatDateShort(selectedSeries.activation_date_central)} on ${formatDateShort(coord.point.date_central)} bankroll ${formatUsd(coord.point.cumulative_bankroll, {
                  minimumFractionDigits: 2,
                })} cumulative net ${formatSignedUsd(coord.point.cumulative_profit, {
                  minimumFractionDigits: 2,
                })}${continuityEnabled ? ` snapshot-only net ${formatSignedUsd(coord.point.snapshot_cumulative_profit, { minimumFractionDigits: 2 })}` : ""}`}
                onPointerEnter={() =>
                  setHoverState({
                    snapshotKey: selectedSeries.snapshot_key,
                    strategy: chartStrategy,
                    date: coord.point.date_central,
                  })
                }
                onPointerDown={() =>
                  setHoverState({
                    snapshotKey: selectedSeries.snapshot_key,
                    strategy: chartStrategy,
                    date: coord.point.date_central,
                  })
                }
                onFocus={() =>
                  setHoverState({
                    snapshotKey: selectedSeries.snapshot_key,
                    strategy: chartStrategy,
                    date: coord.point.date_central,
                  })
                }
                onBlur={() => setHoverState(null)}
              />
              <circle
                cx={coord.x}
                cy={coord.y}
                r={activeCoord?.point.date_central === coord.point.date_central ? 5.6 : 4.1}
                fill={selectedSeries.color}
                stroke="var(--chart-point-stroke)"
                strokeWidth={activeCoord?.point.date_central === coord.point.date_central ? "2.5" : "2"}
                className={activeCoord?.point.date_central === coord.point.date_central ? chartStyles.chartPointActive : undefined}
              />
            </g>
          ))}

          {xTickDates.map((dateKey) => {
            const coord = selectedSeries.coords.find((point) => point.point.date_central === dateKey);
            const x = coord?.x ?? xByDate.get(dateKey) ?? CHART_PAD_LEFT;
            return (
              <text key={dateKey} x={x} y={CHART_HEIGHT - 12} textAnchor="middle" fill="var(--chart-axis)" fontSize="11">
                {formatDateShort(dateKey)}
              </text>
            );
          })}
        </svg>
      </div>

      <div className={chartStyles.chartMeta}>
        <span>
          Start bankroll:{" "}
          <span className={chartStyles.chartMetaStrong}>
            {formatUsd(selectedSeries.starting_bankroll, { minimumFractionDigits: 2 })}
          </span>
        </span>
        <span>
          Selected bankroll:{" "}
          <span className={chartStyles.chartMetaStrong}>
            {formatUsd(selectedSeries.final_point.cumulative_bankroll, { minimumFractionDigits: 2 })}
          </span>
        </span>
        <span>
          {continuityEnabled ? "Account net" : "Selected net"}:{" "}
          <span className={chartStyles.chartMetaStrong}>
            {formatSignedUsd(selectedSeries.display_total_profit, { minimumFractionDigits: 2 })}
          </span>
        </span>
        {continuityEnabled ? (
          <span>
            Snapshot-only net:{" "}
            <span className={chartStyles.chartMetaStrong}>
              {formatSignedUsd(selectedSeries.isolated_total_profit, { minimumFractionDigits: 2 })}
            </span>
          </span>
        ) : null}
        <span>
          Compared through:{" "}
          <span className={chartStyles.chartMetaStrong}>
            {selectedSeries.compared_through_date_central ? formatDateShort(selectedSeries.compared_through_date_central) : "No settled games yet"}
          </span>
        </span>
        <span>
          Basis: <span className={chartStyles.chartMetaStrong}>{continuityEnabled ? "Continuity handoff" : "Independent reset"}</span>
        </span>
        <span>
          Snapshot lines: <span className={chartStyles.chartMetaStrong}>{totalSnapshots}</span>
        </span>
      </div>

      <div className={styles.legendGrid}>
        {geometry.plottedSeries.map((snapshot) => {
          const isSelected = snapshot.snapshot_key === selectedSeries.snapshot_key;
          return (
            <button
              key={snapshot.snapshot_key}
              type="button"
              className={`${styles.legendButton} ${isSelected ? styles.legendButtonActive : ""}`}
              onClick={() => selectSnapshot(snapshot.snapshot_key)}
            >
              <span className={styles.legendSwatch} style={{ backgroundColor: snapshot.color }} aria-hidden="true" />
              <span className={styles.legendText}>
                <strong>Model as of {formatDateShort(snapshot.activation_date_central)}</strong>
                <span>
                  {activeStrategyConfig.label} {formatSignedUsd(snapshot.display_total_profit, { minimumFractionDigits: 2 })} · bankroll{" "}
                  {formatUsd(snapshot.final_point.cumulative_bankroll, { minimumFractionDigits: 2 })}
                </span>
                {continuityEnabled ? (
                  <span>
                    Snapshot-only {formatSignedUsd(snapshot.isolated_total_profit, { minimumFractionDigits: 2 })}
                  </span>
                ) : null}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
