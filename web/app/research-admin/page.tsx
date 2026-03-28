import { Suspense } from "react";
import { notFound } from "next/navigation";
import ResearchAdminView from "@/components/research-admin/ResearchAdminView";
import { isStaticStagingBuild } from "@/lib/static-staging";

export default function ResearchAdminPage() {
  if (isStaticStagingBuild()) {
    notFound();
  }

  return (
    <Suspense fallback={<p className="small">Loading research admin...</p>}>
      <ResearchAdminView />
    </Suspense>
  );
}
