"use client";

import { useSearchParams } from "next/navigation";
import { normalizeLeague } from "@/lib/league";

export function useLeague() {
  const searchParams = useSearchParams();
  return normalizeLeague(searchParams.get("league"));
}
