import { formatUsd } from "@/lib/currency";
import type { BetSizingExplainerGame } from "@/lib/bet-sizing-explainer";
import type { BetSizingStyle } from "@/lib/betting-strategy";
import type { BetSizingPolicyPreview } from "@/lib/bet-sizing-view";
import styles from "./BetSizingStakeFlow.module.css";

type Props = {
  game: BetSizingExplainerGame;
  policy: BetSizingPolicyPreview;
  sizingStyle: BetSizingStyle;
  totalBudget: number;
};

function formatUnits(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)}u`;
}

export default function BetSizingStakeFlow({ game, policy, sizingStyle, totalBudget }: Props) {
  void sizingStyle;

  if (game.requestedStake <= 0 && game.finalStake <= 0) {
    return (
      <article className={`card ${styles.card}`}>
        <div>
          <p className={styles.eyebrow}>Stage 2B</p>
          <h3 className={styles.title}>Stake Staircase</h3>
          <p className="small">This game never reaches the bet-sizing steps because it failed one of the earlier screens.</p>
        </div>
      </article>
    );
  }

  const quotedStake = game.preview.trace.continuousStake;
  const steps = [
    {
      label: "Full Kelly ask",
      value: formatUnits(game.fullKellyUnits),
      note: "The raw bankroll fraction implied by the adjusted edge and the market price.",
    },
    {
      label: `${policy.label} scale`,
      value: formatUnits(game.scaledKellyUnits),
      note: `${policy.fractionalKelly.toFixed(2)}x Kelly scale keeps the profile from betting the raw Kelly amount.`,
    },
    {
      label: "Per-bet cap",
      value: formatUnits(game.cappedKellyUnits),
      note: `No single bet can exceed ${policy.maxBetUnits.toFixed(2)}u under this profile.`,
    },
    {
      label: "Quoted stake",
      value: formatUsd(quotedStake),
      note: "This is the per-game quote before the daily budget decides how much is still available.",
    },
    {
      label: "Final ticket",
      value: formatUsd(game.finalStake),
      note: game.wasTrimmedByBudget
        ? `The daily budget had only ${formatUsd(game.finalStake)} left for this game.`
        : `This is ${Math.round((game.finalStake / totalBudget) * 100)}% of today's maximum budget.`,
    },
  ];

  return (
    <article className={`card ${styles.card}`}>
      <div>
        <p className={styles.eyebrow}>Stage 2B</p>
        <h3 className={styles.title}>Stake Staircase</h3>
        <p className="small">The selected game asks for money in stages, and each stage can only cut the amount down.</p>
      </div>

      <div className={styles.grid}>
        {steps.map((step, index) => (
          <div key={step.label} className={styles.stepCard}>
            <div className={styles.stepTop}>
              <span className={styles.stepIndex}>{index + 1}</span>
              <span className={styles.stepLabel}>{step.label}</span>
            </div>
            <strong className={styles.stepValue}>{step.value}</strong>
            <p className={styles.stepNote}>{step.note}</p>
          </div>
        ))}
      </div>
    </article>
  );
}
