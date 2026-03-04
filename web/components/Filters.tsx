"use client";

import React from "react";

type Props = {
  teams: string[];
  value: string;
  onChange: (v: string) => void;
};

export default function Filters({ teams, value, onChange }: Props) {
  return (
    <div className="card" style={{ display: "flex", gap: 10, alignItems: "center" }}>
      <label htmlFor="team" style={{ fontWeight: 600 }}>
        Team
      </label>
      <select id="team" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">All</option>
        {teams.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>
    </div>
  );
}
