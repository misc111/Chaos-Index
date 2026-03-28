"use client";

import { useSearchParams } from "next/navigation";
import ResearchDeskExperience from "@/components/research-desk/ResearchDeskExperience";
import { normalizeLeague } from "@/lib/league";

export default function ResearchDeskPageClient() {
  const searchParams = useSearchParams();
  return <ResearchDeskExperience league={normalizeLeague(searchParams.get("league"))} />;
}
