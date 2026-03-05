import React from "react";

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

  const cols = Object.keys(rows[0]);

  return (
    <div className="card" style={{ overflowX: "auto" }}>
      <h3 className="title">{title}</h3>
      <table>
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
                <td key={c}>{typeof r[c] === "number" ? r[c].toFixed(4) : String(r[c] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
