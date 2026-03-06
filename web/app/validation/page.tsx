"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ValidationTabs from "@/components/ValidationTabs";
import { normalizeLeague } from "@/lib/league";
import { fetchDashboardJson } from "@/lib/static-staging";

function ValidationPageContent() {
  const [sections, setSections] = useState<Record<string, Record<string, any>[]>>({});
  const searchParams = useSearchParams();
  const league = normalizeLeague(searchParams.get("league"));

  useEffect(() => {
    fetchDashboardJson<{ sections?: Record<string, Record<string, any>[]> }>("validation", "/api/validation", league)
      .then((d) => setSections(d.sections || {}));
  }, [league]);

  return <ValidationTabs sections={sections} />;
}

export default function ValidationPage() {
  return (
    <Suspense fallback={<p className="small">Loading validation...</p>}>
      <ValidationPageContent />
    </Suspense>
  );
}
