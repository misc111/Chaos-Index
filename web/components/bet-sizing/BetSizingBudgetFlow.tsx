import type { CSSProperties } from "react";
import { BetStakeWithIcon, TeamMatchup } from "@/components/TeamWithIcon";
import { formatUsd } from "@/lib/currency";
import type { BetSizingAllocationStep } from "@/lib/bet-sizing-explainer";
import type { LeagueCode } from "@/lib/league";
import styles from "./BetSizingBudgetFlow.module.css";

type Props = {
  league: LeagueCode;
  totalBudget: number;
  allocatedBudget: number;
  remainingBudget: number;
  steps: BetSizingAllocationStep[];
};

function widthStyle(widthPercent: number): CSSProperties {
  return {
    width: `${Math.max(0, Math.min(100, widthPercent))}%`,
  };
}

export default function BetSizingBudgetFlow({
  league,
  totalBudget,
  allocatedBudget,
  remainingBudget,
  steps,
}: Props) {
  if (!steps.length) {
    return (
      <article className={`card ${styles.card}`}>
        <div className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Stage 3</p>
            <h2 className="title">Budget Allocation</h2>
            <p className="small">No game on this slate is currently asking for any of the daily budget.</p>
          </div>
        </div>
      </article>
    );
  }

  return (
    <article className={`card ${styles.card}`}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Stage 3</p>
          <h2 className="title">Watch the Budget Move Across the Slate</h2>
          <p className="small">
            The best surviving bets draw from the same daily pool. Higher-value bets are funded first until the budget runs out.
          </p>
        </div>
        <div className={styles.summaryPills}>
          <span className={styles.summaryPill}>Committed {formatUsd(allocatedBudget)}</span>
          <span className={styles.summaryPill}>Unused {formatUsd(remainingBudget)}</span>
        </div>
      </div>

      <div className={styles.stackBar} aria-label="Daily budget split across funded bets">
        {steps
          .filter((step) => step.finalStake > 0)
          .map((step) => (
            <span
              key={step.gameId}
              className={styles.stackSegment}
              style={widthStyle((step.finalStake / totalBudget) * 100)}
              title={`${step.matchupLabel}: ${formatUsd(step.finalStake)}`}
            />
          ))}
        {remainingBudget > 0 ? (
          <span className={styles.stackRemainder} style={widthStyle((remainingBudget / totalBudget) * 100)} title="Unused budget" />
        ) : null}
      </div>

      <div className={styles.list}>
        {steps.map((step) => {
          const [awayTeam, homeTeam] = step.matchupLabel.split(" at ");
          const usedBefore = Math.max(0, totalBudget - step.budgetBefore);
          const overlayLeft = totalBudget > 0 ? (usedBefore / totalBudget) * 100 : 0;
          const overlayWidth = totalBudget > 0 ? (step.finalStake / totalBudget) * 100 : 0;
          const usedAfter = totalBudget > 0 ? ((totalBudget - step.budgetAfter) / totalBudget) * 100 : 0;

          return (
            <div key={step.gameId} className={styles.row}>
              <div className={styles.rowTop}>
                <div className={styles.rankBadge}>{step.allocationRank}</div>
                <div className={styles.matchupWrap}>
                  <TeamMatchup
                    league={league}
                    awayTeamCode={awayTeam}
                    homeTeamCode={homeTeam}
                    awayLabel={awayTeam}
                    homeLabel={homeTeam}
                    size="sm"
                  />
                </div>
                <BetStakeWithIcon
                  league={league}
                  teamCode={step.team}
                  label={step.team}
                  stake={step.finalStake}
                  size="sm"
                  zeroLabel="$0"
                  className={styles.stakeLabel}
                />
              </div>

              <div className={styles.rowTrack} aria-hidden="true">
                <span className={styles.trackBase} />
                <span className={styles.trackUsed} style={widthStyle(usedAfter)} />
                <span
                  className={`${styles.trackAllocation} ${step.wasTrimmedByBudget ? styles.trackAllocationTrimmed : ""}`}
                  style={
                    {
                      left: `${Math.max(0, Math.min(100, overlayLeft))}%`,
                      width: `${Math.max(0, Math.min(100 - overlayLeft, overlayWidth))}%`,
                    } as CSSProperties
                  }
                />
              </div>

              <div className={styles.rowMeta}>
                <span>Before: {formatUsd(step.budgetBefore)}</span>
                <span>Asked: {formatUsd(step.requestedStake)}</span>
                <span>Funded: {formatUsd(step.finalStake)}</span>
                <span>After: {formatUsd(step.budgetAfter)}</span>
              </div>

              <p className={styles.note}>
                {step.wasTrimmedByBudget
                  ? `${step.note} The full ask was ${formatUsd(step.requestedStake)}, but the daily cap reduced it.`
                  : step.note}
              </p>
            </div>
          );
        })}
      </div>
    </article>
  );
}
