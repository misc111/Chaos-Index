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

function cleanReasonLabel(value: string): string {
  const labels: Record<string, string> = {
    calibration_guardrail: "Calibration",
    minimum_bet_count: "Bet volume",
    minimum_profitable_folds: "Profitable folds",
    missing_ledger_audit: "Missing audit",
    not_full_immutable_pregame_mlb_ledger: "Ledger incomplete",
    not_production_grade: "Not production grade",
    research_backtest_eligible: "Backtest",
    theory_core_candidate: "Theory lane",
    validation_contract_complete: "Validation",
  };
  return labels[value] || titleCaseToken(value);
}

export default function EvidenceStatusCard({
  evidenceStatus,
  promotionEligible,
  productionReady,
  sourceStatus,
}: {
  evidenceStatus?: EvidenceStatusSummary | null;
  promotionEligible?: boolean;
  productionReady?: boolean;
  sourceStatus?: string | null;
}) {
  const evidenceStage = evidenceStatus?.evidence_stage;
  const stageLabel =
    evidenceStage === "fixture_demo_smoke"
      ? "Demo evidence"
      : evidenceStage === "promotion_eligible"
        ? "Promotion eligible"
        : evidenceStage === "production_ready"
          ? "Production ready"
          : evidenceStage === "research_only"
            ? "Research only"
            : promotionEligible
              ? "Promotion eligible"
              : "Blocked";
  const blockedReasons = [
    ...(evidenceStatus?.readiness_blocked_reasons || []),
    ...(evidenceStatus?.blocked_reasons || []),
  ].filter(Boolean);
  const uniqueBlockedReasons = Array.from(new Set(blockedReasons)).slice(0, 4);

  return (
    <section className={`card ${styles.summaryCard}`}>
      <div className={styles.sectionHeader}>
        <div>
          <span className={styles.sectionLabel}>Status</span>
          <h2 className={styles.sectionTitle}>Evidence</h2>
        </div>
      </div>
      <div className={styles.promotionMeta}>
        <span className={`${styles.statusPill} ${productionReady ? styles.statusBet : styles.statusPass}`}>
          {stageLabel}
        </span>
        {sourceStatus ? <span className={styles.pill}>{titleCaseToken(sourceStatus)}</span> : null}
        {evidenceStatus?.full_immutable_pregame_ledger ? <span className={styles.pill}>Full ledger</span> : null}
      </div>
      <p className={styles.copy}>
        {productionReady
          ? "Promotion evidence is ready."
          : promotionEligible
            ? "Evidence is ready for review."
            : "Champion promotion is still blocked."}
      </p>
      {uniqueBlockedReasons.length ? (
        <div className={styles.gateRow}>
          {uniqueBlockedReasons.map((reason) => (
            <span key={reason} className={`${styles.gatePill} ${styles.gateFail}`}>
              {cleanReasonLabel(reason)}
            </span>
          ))}
        </div>
      ) : null}
    </section>
  );
}
