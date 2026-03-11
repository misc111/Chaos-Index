"use client";

import React from "react";
import Charts from "@/components/Charts";
import type { PerformanceScoreRow } from "@/lib/types";
import { displayPredictionModel, orderPredictionModels } from "@/lib/predictions-report";

export default function PerformanceCharts({ rows }: { rows: PerformanceScoreRow[] }) {
  const byModel = rows.reduce<Record<string, PerformanceScoreRow[]>>((acc, r) => {
    acc[r.model_name] = acc[r.model_name] || [];
    acc[r.model_name].push(r);
    return acc;
  }, {});
  const orderedModels = orderPredictionModels(Object.keys(byModel));

  return (
    <div className="grid two">
      {orderedModels.map((model) => (
        <Charts
          key={model}
          title={`${displayPredictionModel(model)} log loss over time`}
          points={(byModel[model] || []).map((p) => ({ x: p.game_date_utc, y: p.log_loss }))}
          color="var(--accent-2)"
        />
      ))}
    </div>
  );
}
