# NBA Research Backtest

Summary
- Seasons included: [20252026]
- Feature pool: research-broad numeric feature pool after leakage bans
- Outer folds / valid days / inner folds / holdout days: 4 / 21 / 2 / 21
- Best candidate model: glm_elastic_net
- Promotion eligible: False

Scorecard
```text
     model_name            strategy  mean_ending_bankroll  median_roi  profit_winning_folds  profitable_folds  mean_log_loss  mean_brier  mean_ece  all_integrity_checks
      glm_ridge capitalPreservation           4894.173171   -0.051077                     2                 2       0.675376    0.236414  0.111913                  True
glm_elastic_net capitalPreservation           4789.953971   -0.174547                     1                 1       0.637238    0.221975  0.107177                  True
      glm_lasso capitalPreservation           4760.725833   -0.193155                     1                 1       0.640257    0.223357  0.102807                  True
    glm_vanilla capitalPreservation           4669.219086   -0.120835                     1                 1       6.092840    0.441014  0.441013                  True
    glm_vanilla        riskAdjusted           4612.219715   -0.057791                     1                 1       6.092840    0.441014  0.441013                  True
    glm_vanilla          aggressive           4464.907384   -0.067886                     1                 1       6.092840    0.441014  0.441013                  True
      glm_lasso        riskAdjusted           3898.727430   -0.183294                     0                 0       0.640257    0.223357  0.102807                  True
      glm_ridge        riskAdjusted           3643.476481   -0.220709                     0                 0       0.675376    0.236414  0.111913                  True
glm_elastic_net        riskAdjusted           3542.918251   -0.297273                     0                 0       0.637238    0.221975  0.107177                  True
      glm_lasso          aggressive           3476.636787   -0.184064                     0                 0       0.640257    0.223357  0.102807                  True
      glm_ridge          aggressive           3116.670985   -0.204019                     0                 0       0.675376    0.236414  0.111913                  True
glm_elastic_net          aggressive           3050.948381   -0.267307                     0                 0       0.637238    0.221975  0.107177                  True
```

Promotion Gate
```json
{"baseline_model": "glm_ridge", "baseline_row": {"all_integrity_checks": true, "bet_count": 114, "embargo_respected": true, "mean_auc": 0.6758040440164823, "mean_brier": 0.23641433099245449, "mean_ece": 0.11191308305789271, "mean_ending_bankroll": 4894.173171193808, "mean_log_loss": 0.6753762736003488, "mean_max_drawdown": 0.07985983106643271, "mean_net_profit": -105.82682880619154, "mean_roi": -0.054999896049908875, "mean_turnover": 0.39749999999999996, "median_roi": -0.0510766554391683, "model_name": "glm_ridge", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 2, "profitable_folds": 2, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_candidate_row": {"all_integrity_checks": true, "bet_count": 75, "embargo_respected": true, "mean_auc": 0.7299542872795598, "mean_brier": 0.22197465926932147, "mean_ece": 0.10717667729097756, "mean_ending_bankroll": 4789.953970725677, "mean_log_loss": 0.6372382268260646, "mean_max_drawdown": 0.07513336061143926, "mean_net_profit": -210.04602927432245, "mean_roi": -0.13935866001151298, "mean_turnover": 0.27325, "median_roi": -0.17454678043380023, "model_name": "glm_elastic_net", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 1, "profitable_folds": 1, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_model": "glm_elastic_net", "checks": {"ece_guardrail": true, "integrity_checks": true, "mean_brier": true, "mean_ending_bankroll": false, "mean_log_loss": true, "median_roi": false, "outer_fold_profit_wins": false}, "eligible": false, "strategy": "capitalPreservation"}
```
