"use client";

import { useEffect, useState } from "react";
import { type LeagueCode } from "@/lib/league";
import { fetchDashboardJson, type StagingDataKey } from "@/lib/static-staging";

export function useDashboardData<T>(
  key: StagingDataKey,
  livePath: string,
  league: LeagueCode,
  emptyValue: T,
  refreshToken?: string | number | null,
  stagingVariant?: string | null
) {
  const [data, setData] = useState<T>(emptyValue);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError("");
      try {
        const payload = await fetchDashboardJson<T>(key, livePath, league, stagingVariant);
        if (!cancelled) {
          setData(payload);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : "Unable to load dashboard data.");
          setData(emptyValue);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [emptyValue, key, league, livePath, refreshToken, stagingVariant]);

  return { data, isLoading, error };
}
