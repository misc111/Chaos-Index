import ModelSprite from "@/components/ModelSprite";
import { displayPredictionModel } from "@/lib/predictions-report";
import styles from "./styles.module.css";

type ModelCard = {
  key: string;
  shortName: string;
  plain: string;
  helps: string;
  fooled: string;
  role: string;
};

type ModelGroup = {
  title: string;
  intro: string;
  models: ModelCard[];
};

const groups: ModelGroup[] = [
  {
    title: "The final desk call",
    intro: "This is the model you usually want first. It listens to the room instead of trusting one voice.",
    models: [
      {
        key: "ensemble",
        shortName: "The table captain",
        role: "Default forecast",
        plain: "Combines the model stack into one calmer probability.",
        helps: "Best when no single model has earned the right to dominate the board.",
        fooled: "Can still share the same blind spot if every model is missing the same baseball context.",
      },
    ],
  },
  {
    title: "The steady core",
    intro: "These are the workhorse actuarial models. They prefer simple relationships and punish noise.",
    models: [
      {
        key: "glm_ridge",
        shortName: "The stabilizer",
        role: "Core GLM",
        plain: "Keeps every input in the conversation, but quiets the loud ones.",
        helps: "Best when a lot of signals are useful but none should get to shout.",
        fooled: "Can keep too many weak signals alive and make the forecast feel safer than it is.",
      },
      {
        key: "glm_elastic_net",
        shortName: "The compromise",
        role: "Core GLM",
        plain: "Blends ridge's patience with lasso's willingness to cut clutter.",
        helps: "Best when related baseball signals travel in packs, like pitching, bullpen, and lineup strength.",
        fooled: "Can mute a real edge if the useful signals are tangled together.",
      },
      {
        key: "glm_lasso",
        shortName: "The pruner",
        role: "Core GLM",
        plain: "Deletes weak inputs so the forecast can focus on the strongest evidence.",
        helps: "Best when the feature set is crowded and the model needs discipline.",
        fooled: "Can cut a subtle signal that only matters in certain matchups.",
      },
      {
        key: "glm_vanilla",
        shortName: "The baseline",
        role: "Core GLM",
        plain: "The simple no-shrinkage version. It shows what the raw GLM wants to say.",
        helps: "Best as a sanity check against the regularized models.",
        fooled: "Most likely to overreact to small samples or noisy features.",
      },
    ],
  },
  {
    title: "Credibility models",
    intro: "These start with another opinion, then ask how much the data should move away from it.",
    models: [
      {
        key: "glm_lasso_market_credibility",
        shortName: "Market-aware credibility",
        role: "Opt-in core",
        plain: "Starts with a clean market opinion and lets lasso decide what the model adds.",
        helps: "Best when the betting market is a useful reference, but not the whole answer.",
        fooled: "Blocked if the market complement is stale, missing, or too close to copying the board.",
      },
      {
        key: "glm_lasso_prior_credibility",
        shortName: "Prior-aware credibility",
        role: "Opt-in core",
        plain: "Starts with a prior model opinion and learns what today's data should change.",
        helps: "Best when there is a trustworthy pregame prior to build from.",
        fooled: "Can inherit old mistakes if the prior is not audited and frozen correctly.",
      },
    ],
  },
  {
    title: "Theory-compatible challengers",
    intro: "These are still actuarial-adjacent, but they are kept out of the default champion lane until evidence catches up.",
    models: [
      {
        key: "gam_spline",
        shortName: "The curve reader",
        role: "Extension",
        plain: "Lets some relationships bend instead of forcing every input into a straight line.",
        helps: "Best when a baseball signal has a sweet spot or threshold.",
        fooled: "Can draw a beautiful curve through noise and call it signal.",
      },
      {
        key: "glmm_logit",
        shortName: "The team-context model",
        role: "Extension",
        plain: "Keeps the main inputs, then gives teams their own background tendencies.",
        helps: "Best when team identity matters beyond the measured features.",
        fooled: "Can confuse team reputation with real current strength.",
      },
      {
        key: "dglm_margin",
        shortName: "The margin translator",
        role: "Extension",
        plain: "Thinks about expected run margin first, then translates that into win chance.",
        helps: "Best when spread shape says more than winner-only data.",
        fooled: "Can drift if the run-margin variance is wrong.",
      },
      {
        key: "goals_poisson",
        shortName: "The run counter",
        role: "Extension",
        plain: "Treats the game like two run-scoring processes and turns that into a win probability.",
        helps: "Best for normal scoring environments where run rates are well behaved.",
        fooled: "Messy pitching changes, weather, or unusual scoring environments can throw it off.",
      },
    ],
  },
  {
    title: "Experimental scouts",
    intro: "These are useful second opinions. They can spot patterns, but they do not get automatic trust.",
    models: [
      {
        key: "mars_hinge",
        shortName: "The bend finder",
        role: "Experimental",
        plain: "Looks for sharp turning points where the relationship changes direction.",
        helps: "Best for discovering thresholds worth testing later.",
        fooled: "Can mistake random wiggles for real turning points.",
      },
      {
        key: "two_stage",
        shortName: "The assembly line",
        role: "Experimental",
        plain: "Solves the problem in two steps instead of one big leap.",
        helps: "Best when one model should create an intermediate baseball view before the final call.",
        fooled: "A bad first stage can poison the second stage.",
      },
      {
        key: "rf",
        shortName: "The voting forest",
        role: "Experimental",
        plain: "Lets many small decision trees vote on the game.",
        helps: "Best for rough interaction hunting.",
        fooled: "Can look confident because many trees repeated the same noisy split.",
      },
      {
        key: "gbdt",
        shortName: "The booster",
        role: "Experimental",
        plain: "Builds a chain of small trees, each trying to fix the last one's misses.",
        helps: "Best when there are repeated patterns the GLMs are too simple to catch.",
        fooled: "Can chase yesterday's mistakes too aggressively.",
      },
      {
        key: "nn_mlp",
        shortName: "The pattern sponge",
        role: "Experimental",
        plain: "A neural model that absorbs many signals and learns flexible combinations.",
        helps: "Best as a broad pattern detector when there is enough clean data.",
        fooled: "Can become a black box that remembers noise instead of baseball truth.",
      },
      {
        key: "bayes_goals",
        shortName: "The uncertainty counter",
        role: "Experimental",
        plain: "A Bayesian run model that tries to carry uncertainty along with the score view.",
        helps: "Best when the size of uncertainty matters as much as the point estimate.",
        fooled: "Can be only as good as the assumptions behind its run process.",
      },
      {
        key: "bayes_bt_state_space",
        shortName: "The form tracker",
        role: "Experimental",
        plain: "Tracks team strength as something that moves over time.",
        helps: "Best when teams are changing and stale season averages lag behind.",
        fooled: "Can overread short hot streaks or cold streaks.",
      },
    ],
  },
];

export default function MeetTheModelsPage() {
  return (
    <div className={styles.page}>
      <section className={styles.intro} aria-labelledby="meet-models-title">
        <div>
          <h2 id="meet-models-title">The model room, without the math fog.</h2>
          <p>
            Each model is a different kind of scout. Some are steady actuaries, some are curve readers, and some are
            experimental scouts that are only here to challenge the room.
          </p>
        </div>
        <div className={styles.legend} aria-label="How to read the model cards">
          <span>Plain idea</span>
          <span>When it helps</span>
          <span>How it gets fooled</span>
        </div>
      </section>

      <div className={styles.groupStack}>
        {groups.map((group) => (
          <section className={styles.group} key={group.title} aria-labelledby={`${group.title}-heading`}>
            <div className={styles.groupHeader}>
              <h3 id={`${group.title}-heading`}>{group.title}</h3>
              <p>{group.intro}</p>
            </div>
            <div className={styles.modelGrid}>
              {group.models.map((model) => (
                <article className={styles.modelCard} key={model.key}>
                  <div className={styles.modelTop}>
                    <ModelSprite model={model.key} className={styles.sprite} />
                    <div>
                      <span className={styles.role}>{model.role}</span>
                      <h4>{displayPredictionModel(model.key)}</h4>
                      <p className={styles.shortName}>{model.shortName}</p>
                    </div>
                  </div>
                  <p className={styles.plain}>{model.plain}</p>
                  <div className={styles.explainGrid}>
                    <div>
                      <span>Helps when</span>
                      <p>{model.helps}</p>
                    </div>
                    <div>
                      <span>Gets fooled by</span>
                      <p>{model.fooled}</p>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
