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
      <div className="nav" style={{ paddingTop: 0 }}>
        {keys.map((k) => (
          <a key={k} onClick={() => setActive(k)} style={{ cursor: "pointer", opacity: active === k ? 1 : 0.6 }}>
            {k}
          </a>
        ))}
      </div>
      <ModelTable title={active} rows={sections[active] || []} />
    </div>
  );
}
