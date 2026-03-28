import { Suspense } from "react";
import ResearchDeskPageClient from "./page-client";

export default function ResearchDeskPage() {
  return (
    <Suspense fallback={<p className="small">Loading research desk...</p>}>
      <ResearchDeskPageClient />
    </Suspense>
  );
}
