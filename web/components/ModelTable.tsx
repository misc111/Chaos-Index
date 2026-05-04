import React from "react";
import type { TableRow } from "@/lib/types";

type Props = {
  title: string;
  rows: TableRow[];
};

function formatCellValue(value: unknown): string {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  if (value == null) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export default function ModelTable({ title, rows }: Props) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card reportCard">
        <div className="reportHeader">
          <div>
            <h3 className="title reportTitle">{title}</h3>
          </div>
          <span className="badge reportBadge">No rows</span>
        </div>
      </div>
    );
  }

  const cols = Object.keys(rows[0]);
  const primaryCol = cols[0] || "";
  const detailCols = primaryCol ? cols.slice(1) : cols;
  const rowLabel = rows.length === 1 ? "1 row" : `${rows.length} rows`;

  return (
    <div className="card reportCard">
      <div className="reportHeader">
        <div>
          <h3 className="title reportTitle">{title}</h3>
        </div>
        <span className="badge reportBadge">{rowLabel}</span>
      </div>
      <div className="tableDesktop reportTableScroll">
        <table className="reportTable">
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              <tr key={idx}>
                {cols.map((c) => (
                  <td key={c}>{formatCellValue(r[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="tableMobile dataCardList">
        {rows.map((row, idx) => (
          <article key={idx} className="dataCard">
            <div className="dataCardHeader">
              <p className="dataCardTitle">
                {primaryCol ? `${primaryCol}: ${formatCellValue(row[primaryCol])}` : `Row ${idx + 1}`}
              </p>
            </div>
            <div className="dataCardGrid">
              {(detailCols.length ? detailCols : cols).map((col) => (
                <div key={col} className="dataField">
                  <span className="dataLabel">{col}</span>
                  <span className="dataValue">{formatCellValue(row[col]) || "—"}</span>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
