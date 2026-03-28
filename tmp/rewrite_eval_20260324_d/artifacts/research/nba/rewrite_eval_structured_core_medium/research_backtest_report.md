# NBA Research Backtest

Summary
- Seasons included: [20252026]
- Feature pool: research-broad numeric feature pool after leakage bans; plus structured GLM experiment `nba_glm_rewrite_v1` from `/Users/davidiruegas/Library/Application Support/SportsModeling/configs/research/nba_glm_rewrite_v1.yaml` (slate `core_market_form`, width variant `medium`; requested=12, usable=12)
- Outer folds / valid days / inner folds / holdout days: 4 / 21 / 2 / 21
- Best candidate model: glm_vanilla
- Promotion eligible: False

Scorecard
```text
     model_name            strategy  mean_ending_bankroll  median_roi  profit_winning_folds  profitable_folds  mean_log_loss  mean_brier  mean_ece  all_integrity_checks
    glm_vanilla capitalPreservation           5083.223074    0.078146                     2                 2       0.623920    0.215092  0.096637                  True
      glm_ridge capitalPreservation           5076.959457    0.068382                     2                 2       0.624519    0.215437  0.094711                  True
glm_elastic_net capitalPreservation           5058.903902    0.040465                     2                 2       0.624617    0.215565  0.096313                  True
      glm_lasso capitalPreservation           5040.709667    0.050921                     2                 2       0.624630    0.215654  0.097328                  True
      glm_lasso        riskAdjusted           2627.065681   -0.418763                     0                 0       0.624630    0.215654  0.097328                  True
glm_elastic_net        riskAdjusted           2615.649971   -0.421988                     0                 0       0.624617    0.215565  0.096313                  True
    glm_vanilla        riskAdjusted           2593.405179   -0.421988                     0                 0       0.623920    0.215092  0.096637                  True
      glm_ridge        riskAdjusted           2550.774971   -0.427077                     0                 0       0.624519    0.215437  0.094711                  True
    glm_vanilla          aggressive           1501.807374   -0.421045                     0                 0       0.623920    0.215092  0.096637                  True
      glm_lasso          aggressive           1472.197043   -0.421641                     0                 0       0.624630    0.215654  0.097328                  True
glm_elastic_net          aggressive           1467.541749   -0.421045                     0                 0       0.624617    0.215565  0.096313                  True
      glm_ridge          aggressive           1434.416749   -0.416883                     0                 0       0.624519    0.215437  0.094711                  True
```

Promotion Gate
```json
{"baseline_model": "glm_ridge", "baseline_row": {"all_integrity_checks": true, "bet_count": 74, "embargo_respected": true, "mean_auc": 0.7400854079027057, "mean_brier": 0.21543680602381662, "mean_ece": 0.09471145514456838, "mean_ending_bankroll": 5076.959457458515, "mean_log_loss": 0.6245187111200906, "mean_max_drawdown": 0.03964030524107218, "mean_net_profit": 76.95945745851533, "mean_roi": 0.0657661003909888, "mean_turnover": 0.27225, "median_roi": 0.06838168485060042, "model_name": "glm_ridge", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 2, "profitable_folds": 2, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_candidate_row": {"all_integrity_checks": true, "bet_count": 75, "embargo_respected": true, "mean_auc": 0.7408557315503034, "mean_brier": 0.21509185652756527, "mean_ece": 0.09663653419636495, "mean_ending_bankroll": 5083.22307401625, "mean_log_loss": 0.6239196199793773, "mean_max_drawdown": 0.040516181396228604, "mean_net_profit": 83.22307401624954, "mean_roi": 0.07064843638684233, "mean_turnover": 0.27575, "median_roi": 0.07814635684230747, "model_name": "glm_vanilla", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 2, "profitable_folds": 2, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_model": "glm_vanilla", "checks": {"ece_guardrail": true, "integrity_checks": true, "mean_brier": true, "mean_ending_bankroll": true, "mean_log_loss": true, "median_roi": true, "outer_fold_profit_wins": false}, "eligible": false, "strategy": "capitalPreservation"}
```
