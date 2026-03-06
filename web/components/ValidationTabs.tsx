"use client";

import React, { useMemo, useState } from "react";
import ModelTable from "@/components/ModelTable";

type Props = {
  sections: Record<string, Record<string, any>[]>;
};

export default function ValidationTabs({ sections }: Props) {
  const keys = useMemo(() => Object.keys(sections), [sections]);
  const [active, setActive] = useState(keys[0] || "");

  if (!keys.length) {
    return <div className="card">No validation artifacts yet.</div>;
  }

  return (
    <div>
      <div className="tabRail" role="tablist" aria-label="Validation sections">
        {keys.map((k) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={active === k}
            className={`tabButton ${active === k ? "active" : ""}`}
            onClick={() => setActive(k)}
          >
            {k}
          </button>
        ))}
      </div>
      <ModelTable title={active} rows={sections[active] || []} />
    </div>
  );
}
