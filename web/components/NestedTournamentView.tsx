"use client";

import React, { useMemo, useState } from "react";
import ModelTable from "@/components/ModelTable";
import type { NestedTournamentResponse, TableRow } from "@/lib/types";

type Props = {
  data: NestedTournamentResponse;
};

function rowsForTarget(rows: TableRow[], target: string): TableRow[] {
  return rows.filter((row) => String(row.target_name || "") === target);
}

function metric(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(4);
  if (value == null || value === "") return "—";
  return String(value);
}

function artifactCount(row: TableRow | undefined): number {
  const artifacts = row?.diagnostic_artifacts;
  if (!artifacts || typeof artifacts !== "object" || Array.isArray(artifacts)) return 0;
  return Object.keys(artifacts).length;
}

function reasonText(row: TableRow): string {
  const summary = row.blocked_reason_summary;
  if (typeof summary === "string" && summary.trim()) return summary;
  const labels = row.gate_reason_labels;
  if (Array.isArray(labels)) return labels.map(String).filter(Boolean).join("; ");
  const reasons = row.gate_reasons;
  if (Array.isArray(reasons)) return reasons.map(String).filter(Boolean).join("; ");
  return "";
}

function titleCaseToken(value: string): string {
  return value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

export default function NestedTournamentView({ data }: Props) {
  const targets = useMemo(() => {
    const names = new Set<string>();
    for (const row of data.target_coverage || []) {
      if (row.target_name) names.add(String(row.target_name));
    }
    for (const row of data.inter_family_leaderboard || []) {
      if (row.target_name) names.add(String(row.target_name));
    }
    return Array.from(names);
  }, [data]);
  const [requestedTarget, setRequestedTarget] = useState("");
  const activeTarget = requestedTarget && targets.includes(requestedTarget) ? requestedTarget : targets[0] || "";

  if (!data.summary || !targets.length) {
    return (
      <div className="card">
        <h3 className="title">Nested Tournament</h3>
        <p className="small">No nested tournament artifact has been published yet.</p>
      </div>
    );
  }

  const coverage = rowsForTarget(data.target_coverage || [], activeTarget);
  const champions = rowsForTarget(data.family_champions || [], activeTarget);
  const finalRows = rowsForTarget(data.inter_family_leaderboard || [], activeTarget);
  const featureCoverage = rowsForTarget(data.feature_coverage_summary || [], activeTarget);
  const blockedChampions = champions.filter((row) => String(row.champion_status || "") !== "shortlist");
  const leader = finalRows[0];
  const evidenceStatus = data.current_best?.evidence_status;
  const latestArtifactRole = String(data.current_best?.latest_artifact_role || "");
  const promotionEligible = data.current_best?.promotion_eligible === true;
  const productionReady = data.current_best?.production_ready === true || (evidenceStatus as TableRow | undefined)?.production_ready === true;
  const evidenceStage = String((evidenceStatus as TableRow | undefined)?.evidence_stage || data.current_best?.evidence_stage || "");

  return (
    <div className="grid">
      <div className="card">
        <div className="tabRail" role="tablist" aria-label="Nested tournament targets">
          {targets.map((target) => (
            <button
              key={target}
              type="button"
              role="tab"
              aria-selected={activeTarget === target}
              className={`tabButton ${activeTarget === target ? "active" : ""}`}
              onClick={() => setRequestedTarget(target)}
            >
              {target}
            </button>
          ))}
        </div>
        <h3 className="title">{activeTarget}</h3>
        <div className="dataCardGrid">
          <div className="dataField">
            <span className="dataLabel">Champion</span>
            <span className="dataValue">{metric(leader?.candidate_key)}</span>
          </div>
          <div className="dataField">
            <span className="dataLabel">Log loss</span>
            <span className="dataValue">{metric(leader?.log_loss)}</span>
          </div>
          <div className="dataField">
            <span className="dataLabel">Brier</span>
            <span className="dataValue">{metric(leader?.brier)}</span>
          </div>
          <div className="dataField">
            <span className="dataLabel">Diagnostics</span>
            <span className="dataValue">{artifactCount(leader)} artifacts</span>
          </div>
          <div className="dataField">
            <span className="dataLabel">Blocked families</span>
            <span className="dataValue">{blockedChampions.length}</span>
          </div>
          <div className="dataField">
            <span className="dataLabel">Evidence status</span>
            <span className="dataValue">
              {evidenceStage ? titleCaseToken(evidenceStage) : productionReady ? "Production Ready" : promotionEligible ? "Promotion Eligible" : "Research Only"}
            </span>
          </div>
          {latestArtifactRole ? (
            <div className="dataField">
              <span className="dataLabel">Latest pointer</span>
              <span className="dataValue">{latestArtifactRole.replace(/_/g, " ")}</span>
            </div>
          ) : null}
        </div>
        {evidenceStatus && typeof evidenceStatus === "object" && !Array.isArray(evidenceStatus) ? (
          <p className="small">
            {String(
              (evidenceStatus as TableRow).pointer_semantics ||
                "Nested tournament evidence is a research recommendation surface, not promotion proof."
            )}
          </p>
        ) : null}
      </div>
      {blockedChampions.length > 0 ? (
        <div className="card">
          <h3 className="title">Validation blockers</h3>
          <div className="tableWrap">
            <table className="dataTable">
              <thead>
                <tr>
                  <th>Family</th>
                  <th>Variant</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {blockedChampions.map((row) => (
                  <tr key={String(row.candidate_key || `${row.model_name}-${row.variant_key}`)}>
                    <td>{metric(row.model_name)}</td>
                    <td>{metric(row.variant_key)}</td>
                    <td>{reasonText(row) || "Validation blocked"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
      <ModelTable title="Target coverage" rows={coverage} />
      <ModelTable title="Feature coverage by split" rows={featureCoverage} />
      <ModelTable title="Family champions" rows={champions} />
      <ModelTable title="Inter-family final holdout" rows={finalRows} />
    </div>
  );
}
