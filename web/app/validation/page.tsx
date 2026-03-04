"use client";

import { useEffect, useState } from "react";
import ValidationTabs from "@/components/ValidationTabs";

export default function ValidationPage() {
  const [sections, setSections] = useState<Record<string, Record<string, any>[]>>({});

  useEffect(() => {
    fetch("/api/validation", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setSections(d.sections || {}));
  }, []);

  return <ValidationTabs sections={sections} />;
}
