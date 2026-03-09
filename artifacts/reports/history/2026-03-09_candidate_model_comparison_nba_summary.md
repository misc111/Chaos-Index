# NBA Candidate Model Comparison

Protocol
- Objective: maximize out-of-sample predictive accuracy for home-win probabilities.
- Guidance followed from the local CAS monograph sections on train/validation/test splitting (4.3), deviance and penalized fit comparisons (6.1-6.2), residual/nonlinearity/stability checks (6.3-6.4), holdout actual-vs-predicted/lift/ROC validation (7.1-7.3), and extension candidates (10.1-10.5).
- Outer split: 40% train, 30% validation, 30% final test, ordered by `start_time_utc`.
- Hyperparameters were tuned with rolling time-series CV inside the fit window for each phase.
- Candidate models were built from the full numeric feature pool after leakage bans, not from the repo's production `glm_feature_subset` or per-model feature map.

Data
- League: NBA
- Historical rows used: 1034
- Train / validation / test rows: 414 / 310 / 310
- Raw candidate features after leakage bans: 163
- Final screened features retained for broad linear models: 131
- Feature-screening counts on final fit window: {'kept': 131, 'dropped': 32}

Validation Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini                                                           params fit_status
Elastic Net GLM  0.665097 0.235779 0.647524 0.076725         0.295047                                             c=0.05; l1_ratio=0.5         ok
      Lasso GLM  0.666580 0.236079 0.651242 0.086476         0.302485                                                            c=0.1         ok
     MARS Hinge  0.680692 0.242514 0.645368 0.106727         0.290737 c=0.25; feature_cap=4; interaction_degree=1; knots_per_feature=4         ok
 Intercept Only  0.685682 0.246276 0.500000 0.005735        -0.085531                                              p_home_win=0.555556         ok
    DGLM Margin  0.686251 0.245661 0.629268 0.116765         0.258536                                      feature_cap=6; iterations=1         ok
      GLM Ridge  0.717489 0.255486 0.640297 0.157340         0.280595                                                           c=0.05         ok
     GAM Spline  0.718402 0.258258 0.611562 0.133296         0.223124                                 c=0.25; feature_cap=3; n_knots=4         ok
     GLMM Logit  0.836458 0.292418 0.523538 0.200777         0.046991                                                   feature_cap=10         ok
    Vanilla GLM  7.397984 0.535483 0.506930 0.535483        -0.051048                                                               {}         ok
```

Final Test Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini                                                           params fit_status
     MARS Hinge  0.675969 0.238805 0.676698 0.111154         0.353397 c=0.25; feature_cap=2; interaction_degree=1; knots_per_feature=4         ok
 Intercept Only  0.700675 0.253740 0.500000 0.061237        -0.039128                                              p_home_win=0.558011         ok
     GAM Spline  0.726349 0.252044 0.643482 0.133118         0.286963                                 c=0.25; feature_cap=3; n_knots=4         ok
      Lasso GLM  0.796471 0.273597 0.622586 0.180110         0.245171                                                            c=0.1         ok
     GLMM Logit  0.911367 0.313860 0.514132 0.234464         0.028388                                                   feature_cap=10         ok
Elastic Net GLM  0.967779 0.305005 0.600857 0.236497         0.201715                                            c=0.05; l1_ratio=0.25         ok
      GLM Ridge  1.616624 0.360232 0.598610 0.330493         0.197219                                                           c=0.05         ok
    DGLM Margin  2.147140 0.341343 0.569451 0.294576         0.137696                                     feature_cap=10; iterations=1         ok
    Vanilla GLM  6.952322 0.503225 0.494214 0.503225        -0.026723                                                               {}         ok
```

Bootstrap Against Final Winner
```text
reference_model comparison_model  delta_log_loss_mean  delta_log_loss_p025  delta_log_loss_p500  delta_log_loss_p975  delta_log_loss_prob_reference_better  delta_brier_mean  delta_brier_p025  delta_brier_p500  delta_brier_p975  delta_brier_prob_reference_better
     mars_hinge      glm_vanilla             6.288077             5.515364             6.308674             7.044305                                 1.000          0.265377          0.200176          0.265293          0.330437                              1.000
     mars_hinge      dglm_margin             1.459118             1.044411             1.451753             1.908992                                 1.000          0.100812          0.058661          0.100016          0.144556                              1.000
     mars_hinge        glm_ridge             0.946549             0.671922             0.945519             1.219381                                 1.000          0.122280          0.067210          0.122222          0.174629                              1.000
     mars_hinge  glm_elastic_net             0.293150             0.161505             0.293601             0.432493                                 1.000          0.066372          0.024516          0.065978          0.107986                              0.998
     mars_hinge       glmm_logit             0.234048             0.131874             0.232208             0.339132                                 1.000          0.074700          0.038734          0.074220          0.112764                              1.000
     mars_hinge        glm_lasso             0.121327             0.026592             0.121758             0.223576                                 0.992          0.035204         -0.001092          0.035482          0.071797                              0.970
     mars_hinge       gam_spline             0.051906            -0.012513             0.050903             0.117923                                 0.937          0.013673         -0.012140          0.013544          0.039071                              0.843
     mars_hinge   intercept_only             0.024615            -0.018689             0.025586             0.063427                                 0.875          0.014902         -0.003327          0.015496          0.031817                              0.944
```

Feature Form Evidence
- Final nonlinearity headline: Nonlinear signal detected in at least one predictor
- Top GAM candidates: none flagged
- Top MARS candidates: away_games_played_prior | diff_player_projection_confidence | dyn_var_diff | diff_availability_uncertainty | away_days_into_season

Recommendation
- Best overall final-holdout model: MARS Hinge (`mars_hinge`)
- Best named candidate on the final holdout: MARS Hinge (`mars_hinge`)
- Final test log loss of the best named candidate: 0.675969
- Final test Brier score of the best named candidate: 0.238805
- Final test AUC of the best named candidate: 0.676698
- Best validation candidate: Elastic Net GLM (`glm_elastic_net`)
- Recommendation: choose MARS Hinge (`mars_hinge`) for the next round if the goal is pure out-of-sample probability accuracy among the tested options.
- Closest challenger on the test set: `glm_vanilla`
- Mean log-loss delta versus winner (challenger minus winner): 6.288077
- 95% bootstrap interval for that log-loss delta: [5.515364, 7.044305]
