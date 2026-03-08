"use client";

import { useSearchParams } from "next/navigation";
import { normalizeBetSizingStyle } from "@/lib/betting-strategy";

export function useBetSizingStyle() {
  const searchParams = useSearchParams();
  return normalizeBetSizingStyle(searchParams.get("sizingStyle"));
}
