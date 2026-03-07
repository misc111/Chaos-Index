"use client";

import React from "react";

type LinePoint = { x: string; y: number };

type Props = {
  title: string;
  points: LinePoint[];
  color?: string;
};

export default function Charts({ title, points, color = "var(--accent)" }: Props) {
  if (!points || points.length === 0) {
    return (
      <div className="card">
        <h3 className="title">{title}</h3>
        <p className="small">No data</p>
      </div>
    );
  }

  const w = 520;
  const h = 180;
  const pad = 26;
  const yVals = points.map((p) => p.y);
  const minY = Math.min(...yVals);
  const maxY = Math.max(...yVals);
  const span = Math.max(maxY - minY, 1e-6);

  const coords = points.map((p, i) => {
    const x = pad + (i / Math.max(points.length - 1, 1)) * (w - 2 * pad);
    const y = h - pad - ((p.y - minY) / span) * (h - 2 * pad);
    return { x, y };
  });

  const d = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.x},${c.y}`).join(" ");

  return (
    <div className="card">
      <h3 className="title">{title}</h3>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "clamp(180px, 42vw, 220px)" }}>
        <rect x="0" y="0" width={w} height={h} rx="12" fill="var(--chart-surface)" />
        <path d={d} fill="none" stroke={color} strokeWidth="2.5" />
      </svg>
    </div>
  );
}
