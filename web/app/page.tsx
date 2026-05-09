/* eslint-disable @next/next/no-img-element */
import ModelSprite from "@/components/ModelSprite";
import { TRAINABLE_MODELS } from "@/lib/generated/model-manifest";
import { MODEL_SPRITES } from "@/lib/model-sprites";
import styles from "./overview.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

type ModelCardIntro = {
  displayName: string;
  cardName?: string;
  cardClass:
    | "Linear Type"
    | "Credibility Type"
    | "Shape Type"
    | "Count Type"
    | "Context Type"
    | "Tree Type"
    | "Neural Type"
    | "Bayesian Type";
  powerName: string;
  power: string;
  attackName: string;
  attack: string;
  flavor: string;
};

const modelCardIntros = {
  glm_ridge: {
    displayName: "Ridge",
    cardClass: "Linear Type",
    powerName: "Noise Guard",
    power: "Quiet the loud inputs without throwing away the whole scouting report.",
    attackName: "Steady Forecast",
    attack: "Turns a crowded board into a calmer pregame probability.",
    flavor: "Can keep weak signals in uniform longer than they deserve.",
  },
  glm_elastic_net: {
    displayName: "Elastic Net",
    cardClass: "Linear Type",
    powerName: "Balanced Grip",
    power: "Keeps related baseball clues together without letting any one clue run wild.",
    attackName: "Signal Blend",
    attack: "Mixes ridge patience with lasso's sharper feature discipline.",
    flavor: "Can soften a real edge when related signals crowd together.",
  },
  glm_lasso: {
    displayName: "Lasso",
    cardClass: "Linear Type",
    powerName: "Feature Trim",
    power: "Cuts clutter so the strongest pregame clues get the spotlight.",
    attackName: "Clean Cut",
    attack: "Drops weak inputs and makes a leaner forecast.",
    flavor: "Subtle matchup context can disappear if it is too quiet.",
  },
  glm_lasso_market_credibility: {
    displayName: "Market Credibility",
    cardName: "Halo Scout",
    cardClass: "Credibility Type",
    powerName: "Price Sense",
    power: "Reads the market as a baseline before deciding what baseball adds.",
    attackName: "Vig-Free Check",
    attack: "Starts from the clean price and looks for justified disagreement.",
    flavor: "The market is a measuring stick, not the answer key.",
  },
  glm_lasso_prior_credibility: {
    displayName: "Prior Credibility",
    cardName: "Memory Pup",
    cardClass: "Credibility Type",
    powerName: "Memory Boost",
    power: "Carries yesterday's useful read into today's game.",
    attackName: "Prior Pull",
    attack: "Starts from a prior forecast and learns the justified adjustment.",
    flavor: "Old mistakes can compound if the prior gets too much trust.",
  },
  glm_vanilla: {
    displayName: "Generalized Linear Model",
    cardName: "Starter Sheet",
    cardClass: "Linear Type",
    powerName: "Clean Read",
    power: "Shows the simple relationship before penalties start negotiating.",
    attackName: "Baseline Swing",
    attack: "Turns pregame features into a straight-ahead probability.",
    flavor: "Small samples and crowded features can push it around.",
  },
  gam_spline: {
    displayName: "Spline",
    cardName: "Curve Charmer",
    cardClass: "Shape Type",
    powerName: "Curve Sense",
    power: "Feels when a baseball relationship needs to bend.",
    attackName: "Bend the Line",
    attack: "Lets a pattern curve when a straight line is too stiff.",
    flavor: "Pretty curves can still be dressed-up noise.",
  },
  glmm_logit: {
    displayName: "Mixed Effects",
    cardClass: "Context Type",
    powerName: "Team Aura",
    power: "Lets team context breathe around the main matchup clues.",
    attackName: "Context Shift",
    attack: "Balances fixed signals with background team tendencies.",
    flavor: "Team identity can blur into reputation if the evidence is stale.",
  },
  dglm_margin: {
    displayName: "Run Margin",
    cardClass: "Shape Type",
    powerName: "Margin Bridge",
    power: "Thinks about the shape of the score before picking a winner.",
    attackName: "Score Swing",
    attack: "Models run margin first, then translates it to win chance.",
    flavor: "Bad variance assumptions can warp the final probability.",
  },
  goals_poisson: {
    displayName: "Run Counter",
    cardClass: "Count Type",
    powerName: "Run Clock",
    power: "Treats each team's scoring like a countable process.",
    attackName: "Score Tally",
    attack: "Turns expected run counts into odds.",
    flavor: "Chaotic pitching, weather, or scoring environments can break rhythm.",
  },
  mars_hinge: {
    displayName: "Threshold Scout",
    cardClass: "Shape Type",
    powerName: "Breakpoint Radar",
    power: "Looks for places where a relationship suddenly changes shape.",
    attackName: "Hinge Hit",
    attack: "Marks thresholds worth testing later.",
    flavor: "Threshold hunting can mistake random wiggles for discovery.",
  },
  two_stage: {
    displayName: "Two Stage",
    cardClass: "Context Type",
    powerName: "Relay Read",
    power: "Builds one baseball read before making the final call.",
    attackName: "Double Play",
    attack: "Passes an intermediate signal into a second decision step.",
    flavor: "A shaky first stage can pass bad information downstream.",
  },
  rf: {
    displayName: "Forest Vote",
    cardClass: "Tree Type",
    powerName: "Many Voices",
    power: "Lets lots of small decision paths vote on messy interactions.",
    attackName: "Tree Chorus",
    attack: "Combines those votes into one matchup read.",
    flavor: "Repeated noisy splits can sound more confident than they are.",
  },
  gbdt: {
    displayName: "Boosted Trees",
    cardClass: "Tree Type",
    powerName: "Comeback Chain",
    power: "Each small tree learns from the last miss.",
    attackName: "Boost Rush",
    attack: "Stacks corrections into a stronger pattern read.",
    flavor: "Can chase yesterday's error too aggressively.",
  },
  nn_mlp: {
    displayName: "Neural Net",
    cardClass: "Neural Type",
    powerName: "Pattern Sponge",
    power: "Absorbs many clues and searches for flexible combinations.",
    attackName: "Hidden Layer",
    attack: "Turns tangled signals into a matchup probability.",
    flavor: "Without enough clean data, memory can masquerade as insight.",
  },
  bayes_goals: {
    displayName: "Bayes Runs",
    cardClass: "Bayesian Type",
    powerName: "Uncertainty Pack",
    power: "Carries uncertainty along with the run-scoring view.",
    attackName: "Posterior Pitch",
    attack: "Updates the run read into a probabilistic game call.",
    flavor: "The score model is only as sturdy as its assumptions.",
  },
  bayes_bt_state_space: {
    displayName: "Form Tracker",
    cardClass: "Bayesian Type",
    powerName: "Moving Form",
    power: "Tracks team strength as something that changes over time.",
    attackName: "State Shift",
    attack: "Turns recent form into a live pregame read.",
    flavor: "Short hot and cold streaks can look too meaningful.",
  },
} satisfies Record<(typeof TRAINABLE_MODELS)[number], ModelCardIntro>;

const modelTypeIconFiles = {
  "Linear Type": "linear.png",
  "Credibility Type": "credibility.png",
  "Shape Type": "shape.png",
  "Count Type": "count.png",
  "Context Type": "context.png",
  "Tree Type": "tree.png",
  "Neural Type": "neural.png",
  "Bayesian Type": "bayesian.png",
} satisfies Record<ModelCardIntro["cardClass"], string>;

const modelTypeCardClasses = {
  "Linear Type": styles.linearTypeCard,
  "Credibility Type": styles.credibilityTypeCard,
  "Shape Type": styles.shapeTypeCard,
  "Count Type": styles.countTypeCard,
  "Context Type": styles.contextTypeCard,
  "Tree Type": styles.treeTypeCard,
  "Neural Type": styles.neuralTypeCard,
  "Bayesian Type": styles.bayesianTypeCard,
} satisfies Record<ModelCardIntro["cardClass"], string>;

export default function HomePage() {
  const heroImage = `${BASE_PATH}/images/chaos-index-hero.png`;

  return (
    <div className={`${styles.page} beauty-home`}>
      <section className={styles.hero} aria-labelledby="home-hero-title">
        <img className={styles.heroImage} src={heroImage} alt="" aria-hidden="true" />
        <div className={styles.heroShade} aria-hidden="true" />
        <div className={styles.heroContent}>
          <p className={styles.eyebrow}>MLB actuarial betting model</p>
          <h1 className={styles.headline} id="home-hero-title">
            Chaos Index
          </h1>
          <p className={styles.heroCopy}>
            Independent baseball probabilities shaped by actuarial discipline, market-aware pricing, and pregame-only
            evidence.
          </p>
        </div>
      </section>

      <section className={styles.intro} aria-label="Model introduction">
        <p className={styles.kicker}>Built by David Iruegas, ACAS</p>
        <h2>Actuarial science, under stadium lights.</h2>
        <p>
          Chaos Index is my scorebook for the weird little arguments baseball starts before first pitch: pitchers,
          prices, lineups, weather, and when the smartest play is no play at all.
        </p>
        <a
          className={styles.textLink}
          href="https://www.linkedin.com/in/david-iruegas/"
          target="_blank"
          rel="noreferrer"
        >
          Connect with David
        </a>
      </section>

      <section className={styles.modelCardBand} aria-labelledby="model-card-title">
        <div className={styles.modelCardHeader}>
          <h2 id="model-card-title">Meet the Models</h2>
          <p>Every forecasting style gets its own card.</p>
        </div>
        <div className={styles.tradingCardGrid}>
          {TRAINABLE_MODELS.map((modelKey) => {
            const sprite = MODEL_SPRITES[modelKey];
            const intro = modelCardIntros[modelKey];
            const cardName = "cardName" in intro ? intro.cardName : undefined;
            const displayCardName = cardName || sprite?.name || intro.displayName;
            const typeIconSrc = `${BASE_PATH}/images/model-sprites/type-icons/${modelTypeIconFiles[intro.cardClass]}`;
            return (
              <article className={`${styles.tradingCard} ${modelTypeCardClasses[intro.cardClass]}`} key={modelKey}>
                <div className={styles.cardHeader}>
                  <h3>{intro.displayName}</h3>
                  <img
                    className={styles.typeIcon}
                    src={typeIconSrc}
                    alt={intro.cardClass}
                    title={intro.cardClass}
                  />
                </div>
                <div className={styles.cardPortrait}>
                  <ModelSprite model={modelKey} className={styles.cardSprite} title={displayCardName} />
                </div>
                <div className={styles.cardSpecies}>
                  <span>{displayCardName}</span>
                  <p>{intro.cardClass}</p>
                </div>
                <div className={styles.cardPower}>
                  <span>Power: {intro.powerName}</span>
                  <p>{intro.power}</p>
                </div>
                <div className={styles.cardAttack}>
                  <span>{intro.attackName}</span>
                  <p>{intro.attack}</p>
                </div>
                <div className={styles.cardFlavor}>
                  <p>{intro.flavor}</p>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
