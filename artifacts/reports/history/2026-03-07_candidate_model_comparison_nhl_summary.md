# NHL Candidate Model Comparison

Protocol
- Objective: maximize out-of-sample predictive accuracy for home-win probabilities.
- Guidance followed from the local CAS monograph sections on train/validation/test splitting (4.3), deviance and penalized fit comparisons (6.1-6.2), residual/nonlinearity/stability checks (6.3-6.4), holdout actual-vs-predicted/lift/ROC validation (7.1-7.3), and extension candidates (10.1-10.5).
- Outer split: 40% train, 30% validation, 30% final test, ordered by `start_time_utc`.
- Hyperparameters were tuned with rolling time-series CV inside the fit window for each phase.
- Candidate models were built from the full numeric feature pool after leakage bans, not from the repo's production `glm_feature_subset` or per-model feature map.

Data
- League: NHL
- Historical rows used: 1122
- Train / validation / test rows: 449 / 336 / 337
- Raw candidate features after leakage bans: 147
- Final screened features retained for broad linear models: 95
- Feature-screening counts on final fit window: {'kept': 95, 'dropped': 52}

Validation Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini                                                          params fit_status
     MARS Hinge  0.681044 0.244979 0.555871 0.055206         0.111883 c=0.1; feature_cap=4; interaction_degree=1; knots_per_feature=3         ok
Elastic Net GLM  0.683105 0.245622 0.555303 0.079847         0.110607                                            c=0.05; l1_ratio=0.5         ok
      Lasso GLM  0.683411 0.245794 0.556899 0.086243         0.113798                                                           c=0.1         ok
     GAM Spline  0.684516 0.246835 0.562465 0.087610         0.124929                                c=0.25; feature_cap=3; n_knots=4         ok
    DGLM Margin  0.684730 0.246102 0.570796 0.058169         0.141591                                    feature_cap=14; iterations=1         ok
 Intercept Only  0.693240 0.250046 0.500000 0.013708        -0.039280                                             p_home_win=0.525612         ok
     GLMM Logit  0.695891 0.251947 0.555126 0.087556         0.110252                                                  feature_cap=10         ok
      GLM Ridge  0.700965 0.251451 0.585295 0.087472         0.170590                                                          c=0.05         ok
    Vanilla GLM  7.154461 0.517856 0.476602 0.517856        -0.060834                                                              {}         ok
```

Final Test Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini                                                          params fit_status
 Intercept Only  0.691465 0.249159 0.500000 0.011412        -0.000212                                             p_home_win=0.519745         ok
     MARS Hinge  0.695019 0.250746 0.498303 0.007027        -0.003536 c=0.1; feature_cap=2; interaction_degree=1; knots_per_feature=3         ok
    DGLM Margin  0.717388 0.260343 0.498055 0.080207        -0.004031                                     feature_cap=6; iterations=1         ok
Elastic Net GLM  0.896561 0.313717 0.567605 0.236040         0.135210                                           c=0.05; l1_ratio=0.75         ok
      Lasso GLM  1.005055 0.319998 0.546213 0.224929         0.092426                                                           c=0.1         ok
     GLMM Logit  1.173478 0.377999 0.491726 0.350336        -0.016689                                                   feature_cap=6         ok
     GAM Spline  1.208011 0.375782 0.522382 0.349109         0.044622                                c=0.25; feature_cap=3; n_knots=4         ok
      GLM Ridge  2.687628 0.454739 0.567428 0.443568         0.134856                                                          c=0.05         ok
    Vanilla GLM  7.051240 0.510385 0.508787 0.510385         0.003253                                                              {}         ok
```

Bootstrap Against Final Winner
```text
reference_model comparison_model  delta_log_loss_mean  delta_log_loss_p025  delta_log_loss_p500  delta_log_loss_p975  delta_log_loss_prob_reference_better  delta_brier_mean  delta_brier_p025  delta_brier_p500  delta_brier_p975  delta_brier_prob_reference_better
 intercept_only      glm_vanilla             6.383146             5.660470             6.397605             7.098199                                 1.000          0.262954          0.209591          0.262768          0.314890                              1.000
 intercept_only        glm_ridge             1.994034             1.586324             1.992239             2.368739                                 1.000          0.205861          0.151267          0.206915          0.255546                              1.000
 intercept_only       gam_spline             0.517464             0.378658             0.517533             0.656034                                 1.000          0.126963          0.086606          0.126828          0.168075                              1.000
 intercept_only       glmm_logit             0.482558             0.358607             0.481154             0.616703                                 1.000          0.129173          0.090014          0.128461          0.171229                              1.000
 intercept_only        glm_lasso             0.314318             0.197266             0.313213             0.432101                                 1.000          0.071167          0.037209          0.071263          0.105181                              1.000
 intercept_only  glm_elastic_net             0.205110             0.118809             0.203652             0.292451                                 1.000          0.064502          0.032277          0.063950          0.095731                              1.000
 intercept_only      dglm_margin             0.025590             0.000563             0.025682             0.053348                                 0.978          0.011033         -0.000602          0.011094          0.023917                              0.970
 intercept_only       mars_hinge             0.003242            -0.008988             0.003105             0.017293                                 0.683          0.001438         -0.004217          0.001362          0.007968                              0.681
```

Feature Form Evidence
- Final nonlinearity headline: Nonlinear signal detected in at least one predictor
- Top GAM candidates: dyn_home_prob | home_r5_shots_against | home_r5_goals_against | home_ewm_goals_against
- Top MARS candidates: elo_home_pre | home_r14_team_save_pct_proxy | home_ewm_shots_against

Recommendation
- Best overall final-holdout model: Intercept Only (`intercept_only`)
- Best named candidate on the final holdout: MARS Hinge (`mars_hinge`)
- Final test log loss of the best named candidate: 0.695019
- Final test Brier score of the best named candidate: 0.250746
- Final test AUC of the best named candidate: 0.498303
- Best validation candidate: MARS Hinge (`mars_hinge`)
- Recommendation: do not switch to any of the tested candidates yet. None beat the intercept-only benchmark on the final holdout under proper scoring rules.
- Among the named candidates, the least-bad test model was MARS Hinge (`mars_hinge`), but it still underperformed the benchmark.
- Closest challenger on the test set: `glm_vanilla`
- Mean log-loss delta versus winner (challenger minus winner): 6.383146
- 95% bootstrap interval for that log-loss delta: [5.660470, 7.098199]
