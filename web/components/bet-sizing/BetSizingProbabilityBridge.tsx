import styles from "./BetSizingProbabilityBridge.module.css";

type Props = {
  marketProbability: number | null;
  referenceProbability: number | null;
  adjustedProbability: number | null;
  rawProbability: number | null;
  expectedValue: number | null;
  edge: number | null;
};

type Marker = {
  key: string;
  label: string;
  value: number;
  tone: "market" | "reference" | "adjusted" | "raw";
};

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatPoints(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pts`;
}

function formatExpectedValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function markerX(value: number, width: number, pad: number): number {
  return pad + value * (width - pad * 2);
}

export default function BetSizingProbabilityBridge({
  marketProbability,
  referenceProbability,
  adjustedProbability,
  rawProbability,
  expectedValue,
  edge,
}: Props) {
  const requiredValues = [marketProbability, referenceProbability, adjustedProbability].every(
    (value) => typeof value === "number" && Number.isFinite(value)
  );

  if (!requiredValues) {
    return (
      <article className={`card ${styles.card}`}>
        <div>
          <p className={styles.eyebrow}>Stage 2A</p>
          <h3 className={styles.title}>Probability Bridge</h3>
          <p className="small">A bridge plot appears here once the app can identify a priced side for the selected game.</p>
        </div>
      </article>
    );
  }

  const markers: Marker[] = [
    {
      key: "market",
      label: "Market fair",
      value: marketProbability || 0,
      tone: "market",
    },
    {
      key: "reference",
      label: "Reference blend",
      value: referenceProbability || 0,
      tone: "reference",
    },
    {
      key: "adjusted",
      label: "Adjusted model",
      value: adjustedProbability || 0,
      tone: "adjusted",
    },
  ];

  if (typeof rawProbability === "number" && Number.isFinite(rawProbability)) {
    markers.push({
      key: "raw",
      label: "Raw model",
      value: rawProbability,
      tone: "raw",
    });
  }

  const width = 760;
  const height = 220;
  const pad = 52;
  const baseY = 122;
  const orderedBridge = markers.filter((marker) => marker.tone !== "raw");
  const bridgePath = orderedBridge
    .map((marker, index) => `${index === 0 ? "M" : "L"}${markerX(marker.value, width, pad)},${baseY}`)
    .join(" ");

  return (
    <article className={`card ${styles.card}`}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Stage 2A</p>
          <h3 className={styles.title}>Probability Bridge</h3>
          <p className="small">The raw model is pulled back toward market reality before any money is assigned.</p>
        </div>
        <div className={styles.metricPills}>
          <span className={styles.metricPill}>Edge {formatPoints(edge)}</span>
          <span className={styles.metricPill}>EV {formatExpectedValue(expectedValue)}</span>
        </div>
      </div>

      <div className={styles.chartWrap}>
        <svg viewBox={`0 0 ${width} ${height}`} className={styles.chart} role="img" aria-label="Probability bridge chart">
          <line x1={pad} y1={baseY} x2={width - pad} y2={baseY} className={styles.axis} />
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const x = markerX(tick, width, pad);
            return (
              <g key={tick}>
                <line x1={x} y1={baseY - 16} x2={x} y2={baseY + 16} className={styles.tick} />
                <text x={x} y={baseY + 36} textAnchor="middle" className={styles.tickLabel}>
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}

          <path d={bridgePath} className={styles.bridgeLine} />

          {markers.map((marker, index) => {
            const x = markerX(marker.value, width, pad);
            const y = marker.tone === "raw" ? baseY - 44 : baseY;
            return (
              <g key={marker.key}>
                {marker.tone === "raw" ? (
                  <line x1={x} y1={baseY - 18} x2={x} y2={y + 10} className={styles.rawGuide} />
                ) : null}
                <circle cx={x} cy={y} r={marker.tone === "adjusted" ? 10 : 8} className={styles[`marker${marker.tone[0].toUpperCase()}${marker.tone.slice(1)}`]} />
                <text
                  x={x}
                  y={marker.tone === "raw" ? y - 16 : baseY - 28 - (index % 2) * 18}
                  textAnchor="middle"
                  className={styles.markerLabel}
                >
                  {marker.label}
                </text>
                <text
                  x={x}
                  y={marker.tone === "raw" ? y + 28 : baseY + 58 + (index % 2) * 16}
                  textAnchor="middle"
                  className={styles.markerValue}
                >
                  {formatPercent(marker.value)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <p className={styles.note}>
        Market fair {formatPercent(marketProbability)} &rarr; reference blend {formatPercent(referenceProbability)} &rarr; adjusted
        model {formatPercent(adjustedProbability)}.
      </p>
    </article>
  );
}
