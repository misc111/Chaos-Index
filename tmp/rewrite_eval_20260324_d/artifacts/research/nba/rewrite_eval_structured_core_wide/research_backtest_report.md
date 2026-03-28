# NBA Research Backtest

Summary
- Seasons included: [20252026]
- Feature pool: research-broad numeric feature pool after leakage bans; plus structured GLM experiment `nba_glm_rewrite_v1` from `/Users/davidiruegas/Library/Application Support/SportsModeling/configs/research/nba_glm_rewrite_v1.yaml` (slate `core_market_form`, width variant `wide`; requested=16, usable=16)
- Outer folds / valid days / inner folds / holdout days: 4 / 21 / 2 / 21
- Best candidate model: glm_vanilla
- Promotion eligible: False

Scorecard
```text
     model_name            strategy  mean_ending_bankroll  median_roi  profit_winning_folds  profitable_folds  mean_log_loss  mean_brier  mean_ece  all_integrity_checks
    glm_vanilla capitalPreservation           5094.347802    0.055155                     2                 2       0.625538    0.215891  0.091524                  True
      glm_ridge capitalPreservation           4993.020747   -0.006215                     2                 2       0.626237    0.216439  0.095812                  True
glm_elastic_net capitalPreservation           4962.247832   -0.024211                     1                 1       0.627879    0.217687  0.098643                  True
      glm_lasso capitalPreservation           4908.944547   -0.152823                     1                 1       0.626284    0.217074  0.094245                  True
      glm_lasso        riskAdjusted           3136.544076   -0.322352                     0                 0       0.626284    0.217074  0.094245                  True
glm_elastic_net        riskAdjusted           2621.321854   -0.418943                     0                 0       0.627879    0.217687  0.098643                  True
    glm_vanilla        riskAdjusted           2577.815681   -0.427017                     0                 0       0.625538    0.215891  0.091524                  True
      glm_ridge        riskAdjusted           2506.019763   -0.453277                     0                 0       0.626237    0.216439  0.095812                  True
      glm_lasso          aggressive           2269.132228   -0.324023                     0                 0       0.626284    0.217074  0.094245                  True
glm_elastic_net          aggressive           1509.465561   -0.415844                     0                 0       0.627879    0.217687  0.098643                  True
    glm_vanilla          aggressive           1460.447043   -0.422364                     0                 0       0.625538    0.215891  0.091524                  True
      glm_ridge          aggressive           1313.651124   -0.448438                     0                 0       0.626237    0.216439  0.095812                  True
```

Promotion Gate
```json
{"baseline_model": "glm_ridge", "baseline_row": {"all_integrity_checks": true, "bet_count": 63, "embargo_respected": true, "mean_auc": 0.7370560869712407, "mean_brier": 0.21643866106231463, "mean_ece": 0.09581203277461928, "mean_ending_bankroll": 4993.020747497753, "mean_log_loss": 0.6262365375046132, "mean_max_drawdown": 0.04465685281631181, "mean_net_profit": -6.979252502247093, "mean_roi": -0.0017085430908020671, "mean_turnover": 0.236, "median_roi": -0.006215002132536757, "model_name": "glm_ridge", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 2, "profitable_folds": 2, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_candidate_row": {"all_integrity_checks": true, "bet_count": 62, "embargo_respected": true, "mean_auc": 0.7376822634298652, "mean_brier": 0.21589106334977742, "mean_ece": 0.09152394193354266, "mean_ending_bankroll": 5094.347801685095, "mean_log_loss": 0.6255382747584342, "mean_max_drawdown": 0.03880752329099865, "mean_net_profit": 94.34780168509485, "mean_roi": 0.08850999081867719, "mean_turnover": 0.2325, "median_roi": 0.05515538275706384, "model_name": "glm_vanilla", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 2, "profitable_folds": 2, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_model": "glm_vanilla", "checks": {"ece_guardrail": true, "integrity_checks": true, "mean_brier": true, "mean_ending_bankroll": true, "mean_log_loss": true, "median_roi": true, "outer_fold_profit_wins": false}, "eligible": false, "strategy": "capitalPreservation"}
```
