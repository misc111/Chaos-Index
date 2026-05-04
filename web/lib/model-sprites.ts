import { MODEL_ALIASES } from "@/lib/generated/model-manifest";

export type ModelSprite = {
  key: string;
  name: string;
  image: string;
  description: string;
};

export const MODEL_SPRITES: Record<string, ModelSprite> = {
  ensemble: {
    key: "ensemble",
    name: "Stacky Ump",
    image: "/images/model-sprites/ensemble.png",
    description: "Combined model verdict scorekeeper.",
  },
  glm_ridge: {
    key: "glm_ridge",
    name: "Ridge Ruler",
    image: "/images/model-sprites/glm_ridge.png",
    description: "Stable ridge-penalized GLM sprite.",
  },
  glm_elastic_net: {
    key: "glm_elastic_net",
    name: "Netty",
    image: "/images/model-sprites/glm_elastic_net.png",
    description: "Balanced elastic-net GLM sprite.",
  },
  glm_lasso: {
    key: "glm_lasso",
    name: "Lasso Loop",
    image: "/images/model-sprites/glm_lasso.png",
    description: "Sparse lasso GLM sprite.",
  },
  glm_lasso_market_credibility: {
    key: "glm_lasso_market_credibility",
    name: "Market Cred",
    image: "/images/model-sprites/glm_lasso_market_credibility.png",
    description: "Market-offset lasso credibility sprite.",
  },
  glm_lasso_prior_credibility: {
    key: "glm_lasso_prior_credibility",
    name: "Prior Cred",
    image: "/images/model-sprites/glm_lasso_prior_credibility.png",
    description: "Prior-offset lasso credibility sprite.",
  },
  glm_vanilla: {
    key: "glm_vanilla",
    name: "Vanilla Logit",
    image: "/images/model-sprites/glm_vanilla.png",
    description: "Plain baseline GLM sprite.",
  },
  gam_spline: {
    key: "gam_spline",
    name: "Spline",
    image: "/images/model-sprites/gam_spline.png",
    description: "Smooth nonlinear GAM sprite.",
  },
  glmm_logit: {
    key: "glmm_logit",
    name: "Mixed Mitt",
    image: "/images/model-sprites/glmm_logit.png",
    description: "Mixed-effects GLMM sprite.",
  },
  dglm_margin: {
    key: "dglm_margin",
    name: "Margin Bridge",
    image: "/images/model-sprites/dglm_margin.png",
    description: "Margin-to-win DGLM sprite.",
  },
  goals_poisson: {
    key: "goals_poisson",
    name: "Run Counter",
    image: "/images/model-sprites/goals_poisson.png",
    description: "Run-count Poisson sprite.",
  },
  mars_hinge: {
    key: "mars_hinge",
    name: "Hinge Scout",
    image: "/images/model-sprites/mars_hinge.png",
    description: "Piecewise hinge challenger sprite.",
  },
  two_stage: {
    key: "two_stage",
    name: "Switchboard",
    image: "/images/model-sprites/two_stage.png",
    description: "Two-stage hybrid challenger sprite.",
  },
  rf: {
    key: "rf",
    name: "Forest Fork",
    image: "/images/model-sprites/rf.png",
    description: "Random forest challenger sprite.",
  },
  gbdt: {
    key: "gbdt",
    name: "Boost Bud",
    image: "/images/model-sprites/gbdt.png",
    description: "Boosted tree challenger sprite.",
  },
  nn_mlp: {
    key: "nn_mlp",
    name: "Nodeball",
    image: "/images/model-sprites/nn_mlp.png",
    description: "Neural MLP challenger sprite.",
  },
  bayes_goals: {
    key: "bayes_goals",
    name: "Bayes Beads",
    image: "/images/model-sprites/bayes_goals.png",
    description: "Bayesian run-count challenger sprite.",
  },
  bayes_bt_state_space: {
    key: "bayes_bt_state_space",
    name: "State Lantern",
    image: "/images/model-sprites/bayes_bt_state_space.png",
    description: "Bayesian state-space challenger sprite.",
  },
};

export function canonicalizeModelSpriteKey(model?: string | null): string {
  const token = String(model || "").trim().toLowerCase();
  return MODEL_ALIASES[token] || token;
}

export function getModelSprite(model?: string | null): ModelSprite | null {
  const key = canonicalizeModelSpriteKey(model);
  return MODEL_SPRITES[key] || null;
}
