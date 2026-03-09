# NBA Candidate Model Comparison

Protocol
- Objective: maximize out-of-sample predictive accuracy for home-win probabilities.
- Guidance followed from the local CAS monograph sections on train/validation/test splitting (4.3), deviance and penalized fit comparisons (6.1-6.2), residual/nonlinearity/stability checks (6.3-6.4), holdout actual-vs-predicted/lift/ROC validation (7.1-7.3), and extension candidates (10.1-10.5).
- Outer split: 40% train, 30% validation, 30% final test, ordered by `start_time_utc`.
- Hyperparameters were tuned with rolling time-series CV inside the fit window for each phase.
- Candidate scope: glm_elastic_net, glm_lasso, glm_ridge, glm_vanilla
- Feature pool: Restricted to the production model feature map for `glm_ridge` after leakage bans (6 raw features).

Data
- League: NBA
- Historical rows used: 1034
- Train / validation / test rows: 414 / 310 / 310
- Raw candidate features after leakage bans: 6
- Final screened features retained for broad linear models: 6
- Feature-screening counts on final fit window: {'kept': 6}

Validation Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini              params fit_status
 Intercept Only  0.685682 0.246276 0.500000 0.005735        -0.085531 p_home_win=0.555556         ok
Elastic Net GLM  0.691228 0.243815 0.643551 0.098024         0.287103 c=0.1; l1_ratio=0.1         ok
      GLM Ridge  0.692202 0.244083 0.643678 0.102870         0.287356               c=0.1         ok
      Lasso GLM  0.702467 0.246879 0.643889 0.104915         0.287779                 c=1         ok
    Vanilla GLM  0.704149 0.247286 0.644143 0.119093         0.288286                  {}         ok
```

Final Test Ranking
```text
   display_name  log_loss    brier      auc      ece  normalized_gini               params fit_status
Elastic Net GLM  0.630706 0.218686 0.722777 0.090308         0.445554 c=0.05; l1_ratio=0.1         ok
      GLM Ridge  0.630844 0.218703 0.722611 0.089156         0.445221               c=0.05         ok
      Lasso GLM  0.631923 0.219254 0.720155 0.072407         0.440310                c=0.1         ok
    Vanilla GLM  0.633024 0.219312 0.718240 0.078571         0.436480                   {}         ok
 Intercept Only  0.700675 0.253740 0.500000 0.061237        -0.039128  p_home_win=0.558011         ok
```

Bootstrap Against Final Winner
```text
reference_model comparison_model  delta_log_loss_mean  delta_log_loss_p025  delta_log_loss_p500  delta_log_loss_p975  delta_log_loss_prob_reference_better  delta_brier_mean  delta_brier_p025  delta_brier_p500  delta_brier_p975  delta_brier_prob_reference_better
glm_elastic_net   intercept_only             0.069471             0.033810             0.070015             0.105140                                 0.999          0.034833          0.019368          0.035019          0.050920                              0.999
glm_elastic_net      glm_vanilla             0.002188            -0.002318             0.002176             0.006591                                 0.835          0.000585         -0.001113          0.000568          0.002238                              0.751
glm_elastic_net        glm_lasso             0.001252            -0.003535             0.001216             0.005981                                 0.699          0.000590         -0.001497          0.000606          0.002716                              0.708
glm_elastic_net        glm_ridge             0.000145            -0.001186             0.000123             0.001521                                 0.586          0.000018         -0.000555          0.000014          0.000587                              0.515
```

Feature Form Evidence
- Final nonlinearity headline: Nonlinear signal detected in at least one predictor
- Top GAM candidates: diff_projected_absence_pressure
- Top MARS candidates: diff_rotation_stability | rest_diff | elo_home_prob

Recommendation
- Best overall final-holdout model: Elastic Net GLM (`glm_elastic_net`)
- Best named candidate on the final holdout: Elastic Net GLM (`glm_elastic_net`)
- Final test log loss of the best named candidate: 0.630706
- Final test Brier score of the best named candidate: 0.218686
- Final test AUC of the best named candidate: 0.722777
- Best validation candidate: Elastic Net GLM (`glm_elastic_net`)
- Recommendation: choose Elastic Net GLM (`glm_elastic_net`) for the next round if the goal is pure out-of-sample probability accuracy among the tested options.
- Closest challenger on the test set: `intercept_only`
- Mean log-loss delta versus winner (challenger minus winner): 0.069471
- 95% bootstrap interval for that log-loss delta: [0.033810, 0.105140]
