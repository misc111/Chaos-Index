# NBA Research Backtest

Summary
- Seasons included: [20252026]
- Feature pool: research-broad numeric feature pool after leakage bans; plus structured GLM experiment `nba_glm_rewrite_v1` from `/Users/davidiruegas/Library/Application Support/SportsModeling/configs/research/nba_glm_rewrite_v1.yaml` (slate `pace_and_pressure`, width variant `medium`; requested=10, usable=10)
- Outer folds / valid days / inner folds / holdout days: 4 / 21 / 2 / 21
- Best candidate model: glm_elastic_net
- Promotion eligible: False

Scorecard
```text
     model_name            strategy  mean_ending_bankroll  median_roi  profit_winning_folds  profitable_folds  mean_log_loss  mean_brier  mean_ece  all_integrity_checks
glm_elastic_net capitalPreservation           4921.097317   -0.054346                     1                 1       0.659455    0.233310  0.084529                  True
      glm_ridge capitalPreservation           4918.597317   -0.054346                     1                 1       0.659676    0.233422  0.085986                  True
      glm_lasso capitalPreservation           4899.847317   -0.054346                     1                 1       0.659845    0.233431  0.086831                  True
    glm_vanilla capitalPreservation           4894.638983   -0.067002                     0                 0       0.663845    0.235252  0.082396                  True
    glm_vanilla        riskAdjusted           3489.255208   -0.277155                     0                 0       0.663845    0.235252  0.082396                  True
glm_elastic_net        riskAdjusted           3212.880208   -0.303782                     0                 0       0.659455    0.233310  0.084529                  True
      glm_lasso        riskAdjusted           3153.442708   -0.318958                     0                 0       0.659845    0.233431  0.086831                  True
      glm_ridge        riskAdjusted           3130.380208   -0.330831                     0                 0       0.659676    0.233422  0.085986                  True
    glm_vanilla          aggressive           2736.265625   -0.272979                     0                 0       0.663845    0.235252  0.082396                  True
glm_elastic_net          aggressive           2566.140625   -0.267485                     0                 0       0.659455    0.233310  0.084529                  True
      glm_lasso          aggressive           2374.828125   -0.301001                     0                 0       0.659845    0.233431  0.086831                  True
      glm_ridge          aggressive           2318.640625   -0.321881                     0                 0       0.659676    0.233422  0.085986                  True
```

Promotion Gate
```json
{"baseline_model": "glm_ridge", "baseline_row": {"all_integrity_checks": true, "bet_count": 42, "embargo_respected": true, "mean_auc": 0.6715273875654714, "mean_brier": 0.23342165457563563, "mean_ece": 0.08598643141274687, "mean_ending_bankroll": 4918.597316766338, "mean_log_loss": 0.6596761492052199, "mean_max_drawdown": 0.032540895970702435, "mean_net_profit": -81.40268323366236, "mean_roi": -0.08445308770683863, "mean_turnover": 0.1575, "median_roi": -0.05434639153034407, "model_name": "glm_ridge", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 1, "profitable_folds": 1, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_candidate_row": {"all_integrity_checks": true, "bet_count": 42, "embargo_respected": true, "mean_auc": 0.6721385169033129, "mean_brier": 0.2333097355548254, "mean_ece": 0.08452852873797281, "mean_ending_bankroll": 4921.097316766338, "mean_log_loss": 0.6594552034618015, "mean_max_drawdown": 0.032044207228980576, "mean_net_profit": -78.90268323366236, "mean_roi": -0.08256607752469095, "mean_turnover": 0.157, "median_roi": -0.05434639153034407, "model_name": "glm_elastic_net", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 1, "profitable_folds": 1, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_model": "glm_elastic_net", "checks": {"ece_guardrail": true, "integrity_checks": true, "mean_brier": true, "mean_ending_bankroll": true, "mean_log_loss": true, "median_roi": false, "outer_fold_profit_wins": false}, "eligible": false, "strategy": "capitalPreservation"}
```
