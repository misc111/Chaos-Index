"use client";

import React from "react";
import Charts from "@/components/Charts";
import { displayPredictionModel } from "@/lib/predictions-report";
import { partitionModelRows } from "@/lib/model-groups";

type Row = { game_date_utc: string; log_loss: number; model_name: string };

export default function PerformanceCharts({ rows }: { rows: Row[] }) {
  const { primaryRows, shadowRows } = partitionModelRows(rows);

  const byModel = (sourceRows: Row[]) =>
    sourceRows.reduce<Record<string, Row[]>>((acc, r) => {
      acc[r.model_name] = acc[r.model_name] || [];
      acc[r.model_name].push(r);
      return acc;
    }, {});

  const primaryByModel = byModel(primaryRows);
  const shadowByModel = byModel(shadowRows);

  const renderSection = (title: string, byModelRows: Record<string, Row[]>, color: string) => {
    const entries = Object.entries(byModelRows);
    if (!entries.length) {
      return null;
    }

    return (
      <section className="card">
        <h3 className="title">{title}</h3>
        <div className="grid two">
          {entries.map(([model, pts]) => (
            <Charts
              key={model}
              title={`${displayPredictionModel(model)} log loss over time`}
              points={pts.map((p) => ({ x: p.game_date_utc, y: p.log_loss }))}
              color={color}
            />
          ))}
        </div>
      </section>
    );
  };

  return (
    <div className="grid">
      {renderSection("Ensemble + Core Models", primaryByModel, "#0f766e")}
      {renderSection("Shadow Models", shadowByModel, "#9f1239")}
    </div>
  );
}
