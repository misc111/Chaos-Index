# MLB Candidate Model Comparison

Protocol
- Objective: maximize out-of-sample probability quality for home-win probabilities under proper scoring rules.
- Guidance followed from the local CAS monograph sections on train/validation/test splitting (4.3), deviance and penalized fit comparisons (6.1-6.2), residual/nonlinearity/stability checks (6.3-6.4), holdout actual-vs-predicted/lift/ROC validation (7.1-7.3), and extension candidates (10.1-10.5).
- Outer split: 40% train, 30% validation, 30% final test, ordered by `start_time_utc`.
- Hyperparameters were tuned with rolling time-series CV inside the fit window for each phase.
- Candidate scope: glm_elastic_net, glm_lasso, glm_ridge, glm_vanilla
- Feature pool: Built from the full numeric feature pool after leakage bans, not from the repo's production `glm_feature_subset` or per-model feature map.

Data
- League: MLB
- Fixture-slice rows used: 480
- Train / validation / test rows: 192 / 144 / 144
- Raw candidate features after leakage bans: 23
- Final screened features retained for broad linear models: 22
- Feature-screening counts on final fit window: {'kept': 22, 'dropped': 1}

Validation Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini                                                                                     params fit_status
Elastic Net GLM  0.667700 0.236511 0.657879 0.095497         0.315759 c=0.0625; l1_ratio=0.05; lambda=16; model_name=glm_elastic_net; penalty_family=elastic_net         ok
      GLM Ridge  0.668204 0.236550 0.659624 0.099385         0.319248             c=0.0625; l1_ratio=None; lambda=16; model_name=glm_ridge; penalty_family=ridge         ok
      Lasso GLM  0.683942 0.242918 0.639465 0.079890         0.278930                c=0.25; l1_ratio=None; lambda=4; model_name=glm_lasso; penalty_family=lasso         ok
 Intercept Only  0.704091 0.255452 0.500000 0.081597         0.039349                                                                        p_home_win=0.546875         ok
    Vanilla GLM  0.732931 0.254483 0.648575 0.175650         0.297151                                                                                         {}         ok
```

Final Test Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini                                                                                      params fit_status
Elastic Net GLM  0.656497 0.232681 0.648825 0.079432         0.297650 c=0.03125; l1_ratio=0.05; lambda=32; model_name=glm_elastic_net; penalty_family=elastic_net         ok
      GLM Ridge  0.658524 0.233666 0.648825 0.082295         0.297650             c=0.03125; l1_ratio=None; lambda=32; model_name=glm_ridge; penalty_family=ridge         ok
      Lasso GLM  0.663950 0.236085 0.635592 0.083063         0.271183                 c=0.25; l1_ratio=None; lambda=4; model_name=glm_lasso; penalty_family=lasso         ok
    Vanilla GLM  0.677017 0.242372 0.623346 0.134638         0.246692                                                                                          {}         ok
 Intercept Only  0.689792 0.248323 0.500000 0.064484        -0.055501                                                                         p_home_win=0.511905         ok
```

Bootstrap Against Final Winner
```text
reference_model comparison_model  delta_log_loss_mean  delta_log_loss_p025  delta_log_loss_p500  delta_log_loss_p975  delta_log_loss_prob_reference_better  delta_brier_mean  delta_brier_p025  delta_brier_p500  delta_brier_p975  delta_brier_prob_reference_better
glm_elastic_net   intercept_only             0.029298            -0.006487             0.026699             0.075453                                  0.91          0.013843         -0.002773          0.012587          0.035821                               0.90
glm_elastic_net      glm_vanilla             0.019372            -0.004436             0.019622             0.051286                                  0.95          0.009328          0.000076          0.008860          0.022418                               0.97
glm_elastic_net        glm_lasso             0.008031            -0.002305             0.007545             0.019671                                  0.92          0.003640         -0.000601          0.003571          0.008881                               0.93
glm_elastic_net        glm_ridge             0.002072            -0.002200             0.001999             0.005875                                  0.86          0.000986         -0.000834          0.000984          0.002709                               0.89
```

Feature Form Evidence
- Final nonlinearity headline: Nonlinear signal detected in at least one predictor
- Top GAM candidates: travel_miles_edge | market_logit_home_prob | market_vig_free_home_prob | starter_quality_edge
- Top MARS candidates: bullpen_fatigue_edge | recent_defense_edge | starter_uncertainty_index

Recommendation
- This comparison is a research screen only; it does not promote a production champion.
- Best final-holdout screen model: Elastic Net GLM (`glm_elastic_net`)
- Best named candidate on the final holdout: Elastic Net GLM (`glm_elastic_net`)
- Final test log loss of the best named candidate: 0.656497
- Final test Brier score of the best named candidate: 0.232681
- Final test AUC of the best named candidate: 0.648825
- Best validation candidate: Elastic Net GLM (`glm_elastic_net`)

Execution Context
- Execution data scope: mlb_fixture_slice
- Source label: deterministic MLB fixture slice generated by scripts/generate_mlb_artifact_slice.py
- Evidence label: bounded MLB slice/fixture evidence only; this is not a production-grade champion promotion.
- Recommendation: advance Elastic Net GLM (`glm_elastic_net`) into research backtest and promotion review if the goal is stronger out-of-sample probability quality among the tested options.
- Closest challenger on the test set: `intercept_only`
- Mean log-loss delta versus winner (challenger minus winner): 0.029298
- 95% bootstrap interval for that log-loss delta: [-0.006487, 0.075453]
- Contract summary: `mlb_fixture_slice_artifact_proof_mlb_comparison_summary.json`
- Recorded recommendation status: `fixture_slice_screen_complete`
- Recommendation rationale: Elastic Net GLM led the named CAS candidates on the fixture-slice final holdout. Rerun on the full immutable pregame MLB ledger before research backtest or promotion review.
