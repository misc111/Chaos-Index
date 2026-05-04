"use client";

import type { EvidenceStatusSummary } from "@/lib/types";
import styles from "./ResearchDeskExperience.module.css";

function titleCaseToken(value: string): string {
  return value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

export default function EvidenceStatusCard({
  evidenceStatus,
  latestArtifactRole,
  promotionEligible,
  productionReady,
  sourceStatus,
}: {
  evidenceStatus?: EvidenceStatusSummary | null;
  latestArtifactRole?: string | null;
  promotionEligible?: boolean;
  productionReady?: boolean;
  sourceStatus?: string | null;
}) {
  const evidenceStage = evidenceStatus?.evidence_stage;
  const stageLabel =
    evidenceStage === "fixture_demo_smoke"
      ? "Fixture/demo/smoke"
      : evidenceStage === "promotion_eligible"
        ? "Promotion-eligible evidence"
        : evidenceStage === "production_ready"
          ? "Production ready"
          : evidenceStage === "research_only"
            ? "Research-only evidence"
            : promotionEligible
              ? "Promotion-eligible evidence"
              : "Blocked from promotion";

  return (
    <section className={`card ${styles.summaryCard}`}>
      <div className={styles.sectionHeader}>
        <div>
          <span className={styles.sectionLabel}>Evidence</span>
          <h2 className={styles.sectionTitle}>Latest artifact status</h2>
        </div>
      </div>
      <div className={styles.promotionMeta}>
        <span className={`${styles.statusPill} ${productionReady ? styles.statusBet : styles.statusPass}`}>
          {stageLabel}
        </span>
        {latestArtifactRole ? <span className={styles.pill}>{titleCaseToken(latestArtifactRole)}</span> : null}
        {sourceStatus ? <span className={styles.pill}>{titleCaseToken(sourceStatus)}</span> : null}
        {evidenceStatus?.fixture_demo_smoke ? <span className={styles.pill}>Fixture/demo/smoke</span> : null}
        {evidenceStatus?.full_immutable_pregame_ledger ? <span className={styles.pill}>Full ledger</span> : null}
        {evidenceStatus?.ledger_audit_present ? <span className={styles.pill}>Ledger audit</span> : null}
      </div>
      <p className={styles.copy}>
        {evidenceStatus?.pointer_semantics ||
          "Latest research recommendations and promotion-eligible evidence are tracked as separate artifact lanes."}
      </p>
      {evidenceStatus?.blocked_reasons?.length ? (
        <div className={styles.gateRow}>
          {evidenceStatus.blocked_reasons.map((reason) => (
            <span key={reason} className={`${styles.gatePill} ${styles.gateFail}`}>
              {titleCaseToken(reason)}
            </span>
          ))}
        </div>
      ) : null}
      {evidenceStatus?.readiness_blocked_reasons?.length ? (
        <div className={styles.gateRow}>
          {evidenceStatus.readiness_blocked_reasons.map((reason) => (
            <span key={reason} className={`${styles.gatePill} ${styles.gateFail}`}>
              {titleCaseToken(reason)}
            </span>
          ))}
        </div>
      ) : null}
    </section>
  );
}
