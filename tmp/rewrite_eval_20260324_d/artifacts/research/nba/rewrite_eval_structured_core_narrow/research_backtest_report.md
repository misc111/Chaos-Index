# NBA Research Backtest

Summary
- Seasons included: [20252026]
- Feature pool: research-broad numeric feature pool after leakage bans; plus structured GLM experiment `nba_glm_rewrite_v1` from `/Users/davidiruegas/Library/Application Support/SportsModeling/configs/research/nba_glm_rewrite_v1.yaml` (slate `core_market_form`, width variant `narrow`; requested=8, usable=8)
- Outer folds / valid days / inner folds / holdout days: 4 / 21 / 2 / 21
- Best candidate model: glm_vanilla
- Promotion eligible: False

Scorecard
```text
     model_name            strategy  mean_ending_bankroll  median_roi  profit_winning_folds  profitable_folds  mean_log_loss  mean_brier  mean_ece  all_integrity_checks
    glm_vanilla capitalPreservation           5203.476506    0.168102                     3                 3       0.618792    0.212894  0.087778                  True
      glm_ridge capitalPreservation           5158.459311    0.110877                     2                 2       0.619803    0.213574  0.087993                  True
      glm_lasso capitalPreservation           5146.771385    0.138911                     3                 3       0.619292    0.213338  0.085384                  True
glm_elastic_net capitalPreservation           5140.914533    0.109513                     2                 2       0.619886    0.213657  0.090410                  True
      glm_ridge        riskAdjusted           2952.832263   -0.364681                     0                 0       0.619803    0.213574  0.087993                  True
glm_elastic_net        riskAdjusted           2921.582263   -0.371925                     0                 0       0.619886    0.213657  0.090410                  True
    glm_vanilla        riskAdjusted           2759.517713   -0.425288                     0                 0       0.618792    0.212894  0.087778                  True
      glm_lasso        riskAdjusted           2710.519763   -0.432532                     0                 0       0.619292    0.213338  0.085384                  True
      glm_ridge          aggressive           1922.088624   -0.365660                     0                 0       0.619803    0.213574  0.087993                  True
glm_elastic_net          aggressive           1878.338624   -0.372483                     0                 0       0.619886    0.213657  0.090410                  True
    glm_vanilla          aggressive           1694.644977   -0.426782                     0                 0       0.618792    0.212894  0.087778                  True
      glm_lasso          aggressive           1597.651124   -0.433606                     0                 0       0.619292    0.213338  0.085384                  True
```

Promotion Gate
```json
{"baseline_model": "glm_ridge", "baseline_row": {"all_integrity_checks": true, "bet_count": 65, "embargo_respected": true, "mean_auc": 0.7478975834012445, "mean_brier": 0.21357401714677268, "mean_ece": 0.0879931394084481, "mean_ending_bankroll": 5158.4593112016155, "mean_log_loss": 0.6198030013706287, "mean_max_drawdown": 0.033570877277698176, "mean_net_profit": 158.45931120161558, "mean_roi": 0.12616344005327715, "mean_turnover": 0.24325, "median_roi": 0.11087738468945799, "model_name": "glm_ridge", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 2, "profitable_folds": 2, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_candidate_row": {"all_integrity_checks": true, "bet_count": 71, "embargo_respected": true, "mean_auc": 0.7490157279330565, "mean_brier": 0.21289380966228819, "mean_ece": 0.08777847600150017, "mean_ending_bankroll": 5203.476506414655, "mean_log_loss": 0.6187924476309514, "mean_max_drawdown": 0.029594613917350496, "mean_net_profit": 203.47650641465435, "mean_roi": 0.15593879100871155, "mean_turnover": 0.266, "median_roi": 0.16810159487848075, "model_name": "glm_vanilla", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 3, "profitable_folds": 3, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_model": "glm_vanilla", "checks": {"ece_guardrail": true, "integrity_checks": true, "mean_brier": true, "mean_ending_bankroll": true, "mean_log_loss": true, "median_roi": true, "outer_fold_profit_wins": false}, "eligible": false, "strategy": "capitalPreservation"}
```
