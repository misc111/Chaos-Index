"use client";

import React from "react";
import Charts from "@/components/Charts";
import type { PerformanceScoreRow } from "@/lib/types";

export default function PerformanceCharts({ rows }: { rows: PerformanceScoreRow[] }) {
  const byModel = rows.reduce<Record<string, PerformanceScoreRow[]>>((acc, r) => {
    acc[r.model_name] = acc[r.model_name] || [];
    acc[r.model_name].push(r);
    return acc;
  }, {});

  return (
    <div className="grid two">
      {Object.entries(byModel).map(([model, pts]) => (
        <Charts
          key={model}
          title={`${model} log loss over time`}
          points={pts.map((p) => ({ x: p.game_date_utc, y: p.log_loss }))}
          color="#8b5cf6"
        />
      ))}
    </div>
  );
}
