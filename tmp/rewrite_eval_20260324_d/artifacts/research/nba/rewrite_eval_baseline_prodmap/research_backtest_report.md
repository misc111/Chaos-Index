# NBA Research Backtest

Summary
- Seasons included: [20252026]
- Feature pool: production model map `glm_ridge`
- Outer folds / valid days / inner folds / holdout days: 4 / 21 / 2 / 21
- Best candidate model: glm_lasso
- Promotion eligible: False

Scorecard
```text
     model_name            strategy  mean_ending_bankroll  median_roi  profit_winning_folds  profitable_folds  mean_log_loss  mean_brier  mean_ece  all_integrity_checks
      glm_lasso capitalPreservation           4845.307661   -0.222882                     1                 1       0.621792    0.215055  0.110293                  True
    glm_vanilla capitalPreservation           4797.513021   -0.184709                     1                 1       0.623021    0.215188  0.099292                  True
      glm_ridge capitalPreservation           4789.238673   -0.222822                     1                 1       0.626209    0.216802  0.096540                  True
glm_elastic_net capitalPreservation           4773.714743   -0.249320                     1                 1       0.623544    0.215784  0.100346                  True
      glm_lasso        riskAdjusted           3439.862031   -0.301395                     0                 0       0.621792    0.215055  0.110293                  True
glm_elastic_net        riskAdjusted           3199.099816   -0.353606                     0                 0       0.623544    0.215784  0.100346                  True
      glm_ridge        riskAdjusted           2804.155371   -0.410187                     0                 0       0.626209    0.216802  0.096540                  True
    glm_vanilla        riskAdjusted           2785.382696   -0.422088                     0                 0       0.623021    0.215188  0.099292                  True
      glm_lasso          aggressive           2621.215612   -0.305629                     0                 0       0.621792    0.215055  0.110293                  True
glm_elastic_net          aggressive           2299.817520   -0.353411                     0                 0       0.623544    0.215784  0.100346                  True
      glm_ridge          aggressive           1695.984187   -0.413264                     0                 0       0.626209    0.216802  0.096540                  True
    glm_vanilla          aggressive           1694.813552   -0.421287                     0                 0       0.623021    0.215188  0.099292                  True
```

Promotion Gate
```json
{"baseline_model": "glm_ridge", "baseline_row": {"all_integrity_checks": true, "bet_count": 65, "embargo_respected": true, "mean_auc": 0.7389128853526605, "mean_brier": 0.21680193079448373, "mean_ece": 0.09654047668158888, "mean_ending_bankroll": 4789.238673254737, "mean_log_loss": 0.6262093983523067, "mean_max_drawdown": 0.0690538320386271, "mean_net_profit": -210.76132674526258, "mean_roi": -0.1819885599756269, "mean_turnover": 0.24275, "median_roi": -0.22282177152186633, "model_name": "glm_ridge", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 1, "profitable_folds": 1, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_candidate_row": {"all_integrity_checks": true, "bet_count": 53, "embargo_respected": true, "mean_auc": 0.7549591947428664, "mean_brier": 0.21505486408995378, "mean_ece": 0.11029346936554386, "mean_ending_bankroll": 4845.307661234571, "mean_log_loss": 0.6217920220125343, "mean_max_drawdown": 0.05735106583403704, "mean_net_profit": -154.69233876542876, "mean_roi": -0.17925982049297493, "mean_turnover": 0.19825, "median_roi": -0.22288156868021125, "model_name": "glm_lasso", "no_missing_results_for_scored": true, "prediction_before_game": true, "profit_winning_folds": 1, "profitable_folds": 1, "strategy": "capitalPreservation", "unique_prediction_keys": true}, "best_model": "glm_lasso", "checks": {"ece_guardrail": false, "integrity_checks": true, "mean_brier": true, "mean_ending_bankroll": true, "mean_log_loss": true, "median_roi": false, "outer_fold_profit_wins": false}, "eligible": false, "strategy": "capitalPreservation"}
```
