"use client";

import type { EvidenceStatusSummary } from "@/lib/types";
import { plainCheckDescription, plainCheckLabel } from "./copy";
import styles from "./ResearchDeskExperience.module.css";

function titleCaseToken(value: string): string {
  return value
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function plainSourceStatus(value: string): string {
  return value === "rejected" ? "Not approved" : titleCaseToken(value);
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
  const approvalLabel = productionReady ? "Yes" : promotionEligible ? "Ready for final review" : "Not yet";
  const approvalText = productionReady
    ? "This model has passed the required checks for full use."
    : promotionEligible
      ? "The evidence is ready for a human review before it becomes official."
      : "The model is useful for testing, but it has not passed enough checks to be treated as official.";
  const blockedReasons = [
    ...(evidenceStatus?.readiness_blocked_reasons || []),
    ...(evidenceStatus?.blocked_reasons || []),
  ].filter(Boolean);
  const uniqueBlockedReasons = Array.from(new Set(blockedReasons)).slice(0, 4);

  return (
    <section className={`card ${styles.summaryCard}`}>
      <div className={styles.sectionHeader}>
        <div>
          <span className={styles.sectionLabel}>Trust check</span>
          <h2 className={styles.sectionTitle}>Can we treat the model as approved?</h2>
        </div>
      </div>
      <div className={styles.answerBlock}>
        <span className={`${styles.answerBadge} ${productionReady ? styles.answerYes : styles.answerNo}`}>
          {approvalLabel}
        </span>
        <p className={styles.copy}>{approvalText}</p>
      </div>
      <div className={styles.statusFacts}>
        {evidenceStage ? (
          <span>
            Review level: <strong>{evidenceStage === "research_only" ? "Still being tested" : titleCaseToken(evidenceStage)}</strong>
          </span>
        ) : null}
        {sourceStatus ? (
          <span>
            Latest test result: <strong>{plainSourceStatus(sourceStatus)}</strong>
          </span>
        ) : null}
        {evidenceStatus?.full_immutable_pregame_ledger ? (
          <span>
            Pregame log: <strong>Complete</strong>
          </span>
        ) : null}
      </div>
      {uniqueBlockedReasons.length ? (
        <div className={styles.checkList} aria-label="Checks that still need work">
          {uniqueBlockedReasons.map((reason) => (
            <div key={reason} className={styles.checkItem}>
              <span className={styles.checkStatus}>Needs work</span>
              <div>
                <strong>{plainCheckLabel(reason)}</strong>
                <p>{plainCheckDescription(reason)}</p>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
