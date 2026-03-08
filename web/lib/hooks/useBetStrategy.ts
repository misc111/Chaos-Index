"use client";

import { useSearchParams } from "next/navigation";
import { normalizeBetStrategy } from "@/lib/betting-strategy";

export function useBetStrategy() {
  const searchParams = useSearchParams();
  return normalizeBetStrategy(searchParams.get("strategy"));
}
