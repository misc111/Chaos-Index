import React from "react";
import { partitionModelRows } from "@/lib/model-groups";
import { displayPredictionModel } from "@/lib/predictions-report";

type Props = {
  title: string;
  rows: Record<string, any>[];
};

export default function ModelTable({ title, rows }: Props) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card">
        <h3 className="title">{title}</h3>
        <p className="small">No rows available</p>
      </div>
    );
  }

  const renderTable = (tableRows: Record<string, any>[], heading?: string) => {
    if (!tableRows.length) {
      return null;
    }

    const cols = Object.keys(tableRows[0]);

    return (
      <section style={{ display: "grid", gap: 10 }}>
        {heading ? <p className="small" style={{ margin: 0, fontWeight: 700 }}>{heading}</p> : null}
        <table>
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r, idx) => (
              <tr key={idx}>
                {cols.map((c) => {
                  const value = r[c];
                  const rendered =
                    c === "model_name" && typeof value === "string"
                      ? displayPredictionModel(value)
                      : typeof value === "number"
                        ? value.toFixed(4)
                        : String(value ?? "");

                  return <td key={c}>{rendered}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    );
  };

  const hasModelColumn = rows.some((row) => typeof row.model_name === "string");
  const { primaryRows, shadowRows, otherRows } = hasModelColumn ? partitionModelRows(rows) : { primaryRows: rows, shadowRows: [], otherRows: [] };

  return (
    <div className="card" style={{ overflowX: "auto", display: "grid", gap: 16 }}>
      <h3 className="title">{title}</h3>
      {renderTable(primaryRows, hasModelColumn ? "Ensemble + core models" : undefined)}
      {renderTable(shadowRows, shadowRows.length ? "Shadow models" : undefined)}
      {renderTable(otherRows, otherRows.length ? "Other rows" : undefined)}
    </div>
  );
}
