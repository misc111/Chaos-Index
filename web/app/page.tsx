/* eslint-disable @next/next/no-img-element */
import { loadEvidenceStatus } from "@/app/api/research-desk/route-support";
import ModelSprite from "@/components/ModelSprite";
import { TRAINABLE_MODELS } from "@/lib/generated/model-manifest";
import { buildHomeModelEvidenceBadge, indexCurrentBestTopModels } from "@/lib/home-model-evidence";
import { MODEL_SPRITES } from "@/lib/model-sprites";
import styles from "./overview.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

type ModelCardIntro = {
  displayName: string;
  move: string;
  watch: string;
};

const modelCardIntros = {
  glm_ridge: {
    displayName: "Ridge",
    move: "Quiet the noisy inputs without throwing away the whole scouting report.",
    watch: "Can keep weak signals in uniform longer than they deserve.",
  },
  glm_elastic_net: {
    displayName: "Elastic Net",
    move: "Blend ridge stability with lasso's sharper feature discipline.",
    watch: "Can soften a real edge when related signals crowd together.",
  },
  glm_lasso: {
    displayName: "Lasso",
    move: "Trim the feature set down to the strongest pregame clues.",
    watch: "Subtle matchup context can disappear if it is too quiet.",
  },
  glm_lasso_market_credibility: {
    displayName: "Market Credibility",
    move: "Start from the vig-free market and ask what MLB features add.",
    watch: "Needs the market offset to stay a benchmark, not an answer key.",
  },
  glm_lasso_prior_credibility: {
    displayName: "Prior Credibility",
    move: "Start from a prior forecast and learn the justified adjustment.",
    watch: "Inherited prior mistakes must be audited before trust compounds.",
  },
  glm_vanilla: {
    displayName: "Vanilla",
    move: "Show the raw GLM read before penalties start negotiating.",
    watch: "Small samples and crowded features can push it around.",
  },
  gam_spline: {
    displayName: "Spline",
    move: "Let a baseball relationship bend when a straight line is too stiff.",
    watch: "Pretty curves can still be dressed-up noise.",
  },
  glmm_logit: {
    displayName: "Mixed Effects",
    move: "Keep fixed matchup signals while letting team context breathe.",
    watch: "Team identity can blur into reputation if the evidence is stale.",
  },
  dglm_margin: {
    displayName: "Run Margin",
    move: "Model the run-margin shape before translating it to win chance.",
    watch: "Bad variance assumptions can warp the final probability.",
  },
  goals_poisson: {
    displayName: "Run Counter",
    move: "Count run-scoring processes and turn the scoreboard into odds.",
    watch: "Chaotic pitching, weather, or scoring environments can break rhythm.",
  },
  mars_hinge: {
    displayName: "Threshold Scout",
    move: "Scout for thresholds where a relationship suddenly changes shape.",
    watch: "Threshold hunting can mistake random wiggles for discovery.",
  },
  two_stage: {
    displayName: "Two Stage",
    move: "Build an intermediate baseball read before making the final call.",
    watch: "A shaky first stage can pass bad information downstream.",
  },
  rf: {
    displayName: "Forest Vote",
    move: "Let many small trees vote on messy interaction patterns.",
    watch: "Repeated noisy splits can sound more confident than they are.",
  },
  gbdt: {
    displayName: "Boosted Trees",
    move: "Boost a chain of trees, each learning from the last miss.",
    watch: "Can chase yesterday's error too aggressively.",
  },
  nn_mlp: {
    displayName: "Neural Net",
    move: "Absorb many signals and hunt flexible combinations.",
    watch: "Without enough clean data, memory can masquerade as insight.",
  },
  bayes_goals: {
    displayName: "Bayes Runs",
    move: "Carry uncertainty through a Bayesian run-scoring view.",
    watch: "The score model is only as sturdy as its assumptions.",
  },
  bayes_bt_state_space: {
    displayName: "Form Tracker",
    move: "Track team strength as a form line that moves over time.",
    watch: "Short hot and cold streaks can look too meaningful.",
  },
} satisfies Record<(typeof TRAINABLE_MODELS)[number], ModelCardIntro>;

export default function HomePage() {
  const heroImage = `${BASE_PATH}/images/chaos-index-hero.png`;
  const evidence = loadEvidenceStatus("MLB");
  const topModels = indexCurrentBestTopModels(evidence.current_best_top_models);

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
          {TRAINABLE_MODELS.map((modelKey, index) => {
            const sprite = MODEL_SPRITES[modelKey];
            const intro = modelCardIntros[modelKey];
            const evidenceNote = buildHomeModelEvidenceBadge(modelKey, evidence, topModels);
            return (
              <article className={styles.tradingCard} key={modelKey}>
                <div className={styles.cardChrome}>
                  <span>No. {String(index + 1).padStart(3, "0")}</span>
                </div>
                <div className={styles.cardPortrait}>
                  <ModelSprite model={modelKey} className={styles.cardSprite} />
                </div>
                <div className={styles.cardIdentity}>
                  <p>{sprite?.name || intro.displayName}</p>
                  <h3>{intro.displayName}</h3>
                </div>
                <div className={`${styles.cardEvidence} ${styles[evidenceNote.tone]}`}>
                  <span>So far</span>
                  <p>{evidenceNote.reason}</p>
                </div>
                <div className={styles.cardMove}>
                  <span>Signature move</span>
                  <p>{intro.move}</p>
                </div>
                <div className={styles.cardWatch}>
                  <span>Watch for</span>
                  <p>{intro.watch}</p>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
