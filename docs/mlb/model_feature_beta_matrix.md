# MLB Model Feature/Beta Matrix

Generated at: `2026-04-23T17:52:52+00:00`

Regenerate with:
`python3 scripts/update_model_feature_beta_matrix.py --config configs/mlb.yaml`

This file separates the production theory-core fit from the latest candidate-comparison fit so the same model key does not silently refer to two different feature sets.

## Production Theory-Core Snapshot

Source: current processed MLB feature dataset plus the committed production feature map.

| Display Name                 | Model                        | Lane | Selected Features | Coefficient Rows | Notes                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------- | ---------------------------- | ---- | ----------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| glm_lasso_market_credibility | glm_lasso_market_credibility | core | 0                 | 0                | Skipped production fit because required complement column `market_offset_logit` is missing from the current processed dataset. Missing mapped features: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_lineup_talent, park_run_effect, home_field_advantage.                                                                                                 |
| glm_lasso_prior_credibility  | glm_lasso_prior_credibility  | core | 0                 | 0                | Skipped production fit because required complement column `prior_model_offset_logit` is missing from the current processed dataset. Missing mapped features: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_lineup_talent, park_run_effect, home_field_advantage.                                                                                            |
| glm_ridge                    | glm_ridge                    | core | 2                 | 2                | Production fit on current processed MLB features using the committed model feature map (2 available features). Missing mapped features in the current dataset: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_bullpen_fatigue, diff_lineup_talent, lineup_stability_diff, park_run_effect, park_weather_run_effect, home_field_advantage.                    |
| glm_elastic_net              | glm_elastic_net              | core | 2                 | 2                | Production fit on current processed MLB features using the committed model feature map (2 available features). Missing mapped features in the current dataset: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_bullpen_fatigue, diff_lineup_talent, lineup_stability_diff, park_run_effect, park_weather_run_effect, home_field_advantage, umpire_run_effect. |
| glm_lasso                    | glm_lasso                    | core | 2                 | 2                | Production fit on current processed MLB features using the committed model feature map (2 available features). Missing mapped features in the current dataset: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_lineup_talent, park_run_effect, home_field_advantage.                                                                                          |

### Feature Matrix

| Feature     | glm_lasso_market_credibility | glm_lasso_prior_credibility | glm_ridge | glm_elastic_net | glm_lasso |
| ----------- | ---------------------------- | --------------------------- | --------- | --------------- | --------- |
| rest_diff   |                              |                             | -0.0043   | 0.0000          | 0.0000    |
| travel_diff |                              |                             | 0.0000    | 0.0000          | 0.0000    |

### Coefficient Detail

### glm_lasso_market_credibility (`glm_lasso_market_credibility`)

Skipped production fit because required complement column `market_offset_logit` is missing from the current processed dataset. Missing mapped features: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_lineup_talent, park_run_effect, home_field_advantage.

No coefficient rows are available for this model snapshot.

### glm_lasso_prior_credibility (`glm_lasso_prior_credibility`)

Skipped production fit because required complement column `prior_model_offset_logit` is missing from the current processed dataset. Missing mapped features: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_lineup_talent, park_run_effect, home_field_advantage.

No coefficient rows are available for this model snapshot.

### glm_ridge (`glm_ridge`)

Production fit on current processed MLB features using the committed model feature map (2 available features). Missing mapped features in the current dataset: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_bullpen_fatigue, diff_lineup_talent, lineup_stability_diff, park_run_effect, park_weather_run_effect, home_field_advantage.

| Term        | Base Feature | Role   | Beta    | Scale            | Note |
| ----------- | ------------ | ------ | ------- | ---------------- | ---- |
| rest_diff   | rest_diff    | driver | -0.0043 | original_feature |      |
| travel_diff | travel_diff  | driver | 0.0000  | original_feature |      |

### glm_elastic_net (`glm_elastic_net`)

Production fit on current processed MLB features using the committed model feature map (2 available features). Missing mapped features in the current dataset: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_bullpen_fatigue, diff_lineup_talent, lineup_stability_diff, park_run_effect, park_weather_run_effect, home_field_advantage, umpire_run_effect.

| Term        | Base Feature | Role   | Beta   | Scale            | Note |
| ----------- | ------------ | ------ | ------ | ---------------- | ---- |
| rest_diff   | rest_diff    | driver | 0.0000 | original_feature |      |
| travel_diff | travel_diff  | driver | 0.0000 | original_feature |      |

### glm_lasso (`glm_lasso`)

Production fit on current processed MLB features using the committed model feature map (2 available features). Missing mapped features in the current dataset: diff_starting_pitcher_quality, starting_pitcher_hand_matchup, diff_bullpen_quality, diff_lineup_talent, park_run_effect, home_field_advantage.

| Term        | Base Feature | Role   | Beta   | Scale            | Note |
| ----------- | ------------ | ------ | ------ | ---------------- | ---- |
| rest_diff   | rest_diff    | driver | 0.0000 | original_feature |      |
| travel_diff | travel_diff  | driver | 0.0000 | original_feature |      |

## Candidate Comparison Snapshot

Source: latest candidate comparison manifest plus a reconstruction of the latest comparison fit window and selected hyperparameters.

| Display Name    | Model           | Lane         | Selected Features | Coefficient Rows | Notes                                                                                                                                                                                                                                             |
| --------------- | --------------- | ------------ | ----------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GLM Ridge       | glm_ridge       | core         | 50                | 50               | Comparison fit with original-scale linear coefficients. Current comparison params: c=0.015625; l1_ratio=None; lambda=64; model_name=glm_ridge; penalty_family=ridge. Fit on the latest comparison fit window before the final holdout.            |
| Elastic Net GLM | glm_elastic_net | core         | 50                | 50               | Comparison fit with original-scale linear coefficients. Current comparison params: c=0.03125; l1_ratio=0.85; lambda=32; model_name=glm_elastic_net; penalty_family=elastic_net. Fit on the latest comparison fit window before the final holdout. |
| Lasso GLM       | glm_lasso       | core         | 50                | 50               | Comparison fit with original-scale linear coefficients. Current comparison params: c=0.0416667; l1_ratio=None; lambda=24; model_name=glm_lasso; penalty_family=lasso. Fit on the latest comparison fit window before the final holdout.           |
| Vanilla GLM     | glm_vanilla     | core         | 50                | 50               | Comparison fit with original-scale GLM coefficients. Current comparison params: {}. Fit on the latest comparison fit window before the final holdout.                                                                                             |
| GLMM Logit      | glmm_logit      | extension    | 6                 | 6                | Fixed-effect coefficients only; team random-effect terms are omitted from the matrix. Current comparison params: feature_cap=6. Fit on the latest comparison fit window before the final holdout.                                                 |
| DGLM Margin     | dglm_margin     | extension    | 6                 | 12               | Mean and dispersion coefficients are both shown for each selected feature. Current comparison params: feature_cap=6; iterations=1. Fit on the latest comparison fit window before the final holdout.                                              |
| GAM Spline      | gam_spline      | extension    | 8                 | 23               | Spline rows are basis-term coefficients, not a single beta on the raw predictor scale. Current comparison params: c=0.5; feature_cap=3; n_knots=5. Fit on the latest comparison fit window before the final holdout.                              |
| MARS Hinge      | mars_hinge      | experimental | 5                 | 10               | Hinge rows are basis-term coefficients on the MARS proxy basis. Current comparison params: c=0.1; feature_cap=2; interaction_degree=1; knots_per_feature=3. Fit on the latest comparison fit window before the final holdout.                     |

### Base-Feature Selection Matrix

| Feature                    | glm_ridge | glm_elastic_net | glm_lasso | glm_vanilla | glmm_logit | dglm_margin | gam_spline | mars_hinge |
| -------------------------- | --------- | --------------- | --------- | ----------- | ---------- | ----------- | ---------- | ---------- |
| home_starter_runs_allowed  | -0.5548   | -1.0651         | -1.4934   | -27.0669    | -3.4840    | mean,disp   | spline     | -2.2566    |
| away_starter_runs_allowed  | 0.5212    | 1.0287          | 1.4623    | 27.0951     | 3.4632     | mean,disp   | spline     | 2.2330     |
| elo_home_prob              | -0.0146   | 0.0000          | 0.0000    | 28.9304     |            |             |            |            |
| elo_home_pre               | 0.0004    | 0.0000          | 0.0000    | -0.0200     |            |             |            |            |
| dyn_home_prob              | 0.2482    | 0.0000          | 0.0000    | 2.4218      |            |             |            |            |
| home_ewm_run_diff          | 0.0114    | 0.0000          | 0.0000    | -1.2911     |            |             |            |            |
| home_r14_run_diff          | 0.0077    | 0.0000          | 0.0000    | 0.0930      |            |             |            |            |
| away_r5_runs_for           | -0.0532   | 0.0000          | 0.0000    | -0.0447     | -0.0473    | mean,disp   | -0.0384    |            |
| dyn_home_mean              | 0.1081    | 0.0000          | 0.0000    | 0.5404      |            |             | spline     |            |
| elo_home_prob_hinge_052    | -0.1236   | 0.0000          | 0.0000    | -0.6180     |            |             |            |            |
| away_r5_run_diff           | -0.0191   | 0.0000          | 0.0000    | -0.0658     |            |             |            |            |
| away_ewm_run_diff          | 0.0014    | 0.0000          | 0.0000    | 11.6217     |            |             |            |            |
| home_r5_run_diff           | -0.0079   | 0.0000          | 0.0000    | -0.0576     |            |             |            |            |
| diff_form_win_rate         | -0.0817   | 0.0000          | 0.0000    | 10.0700     |            |             |            | hinge      |
| home_win_rate_ewm          | 0.1062    | 0.0000          | 0.0000    | -12.4059    |            |             |            |            |
| home_r14_runs_against      | -0.0079   | 0.0000          | 0.0000    | -0.2045     |            |             |            |            |
| home_ewm_runs_against      | 0.0113    | 0.0000          | 0.0000    | -1.2307     |            |             |            |            |
| dyn_away_mean              | -0.0478   | 0.0000          | 0.0000    | -0.1494     |            |             |            |            |
| away_r14_run_diff          | -0.0017   | 0.0000          | 0.0000    | 29.6652     |            |             |            |            |
| home_ewm_runs_for          | 0.0374    | 0.0000          | 0.0000    | 1.4804      |            |             |            |            |
| away_ewm_runs_for          | 0.0292    | 0.0000          | 0.0000    | -11.7112    |            |             |            |            |
| home_r14_runs_for          | 0.0072    | 0.0000          | 0.0000    | -0.0454     |            |             |            |            |
| home_r5_runs_against       | 0.0173    | 0.0000          | 0.0000    | 0.1108      |            |             |            |            |
| elo_away_pre               | 0.0003    | 0.0000          | 0.0000    | 0.0178      |            |             |            |            |
| away_r14_runs_against      | 0.0114    | 0.0000          | 0.0000    | 29.5393     |            |             |            |            |
| away_win_rate_ewm          | 0.3351    | 0.0000          | 0.0000    | 11.9795     |            |             |            |            |
| away_ewm_runs_against      | 0.0269    | 0.0000          | 0.0000    | 11.5555     |            |             |            |            |
| home_r5_runs_for           | 0.0023    | 0.0000          | 0.0000    | 0.0010      |            |             |            |            |
| away_r14_runs_for          | 0.0083    | 0.0000          | 0.0000    | -29.3684    |            |             |            |            |
| away_3in4                  | -0.1202   | 0.0000          | 0.0000    | -1.4604     |            |             |            |            |
| home_tz_change             | -0.0529   | 0.0000          | 0.0000    | -0.1516     |            |             |            |            |
| away_r5_runs_against       | -0.0148   | 0.0000          | 0.0000    | 0.0906      |            |             |            |            |
| home_3in4                  | 0.0201    | 0.0000          | 0.0000    | 1.6311      |            |             |            |            |
| away_travel_miles          | -0.0001   | 0.0000          | 0.0000    | 0.0002      |            |             |            |            |
| home_4in6                  | -0.0083   | 0.0000          | 0.0000    | -1.5739     |            |             |            |            |
| home_rest_days_x           | 0.0374    | 0.0000          | 0.0000    | 0.1780      |            |             |            |            |
| away_rest_days_x           | 0.0420    | 0.0000          | 0.0000    | 0.0789      |            |             |            |            |
| home_travel_miles          | -0.0001   | 0.0000          | 0.0000    | -0.0001     |            |             |            |            |
| away_tz_change             | -0.0190   | 0.0000          | 0.0000    | 0.2538      |            |             |            |            |
| away_b2b_x                 | 0.0329    | 0.0000          | 0.0000    | 0.4614      |            |             |            |            |
| home_b2b_x                 | -0.0013   | 0.0000          | 0.0000    | -0.2457     |            |             |            |            |
| travel_diff                | 0.0000    | 0.0000          | 0.0000    | -0.0002     |            |             |            |            |
| away_4in6                  | 0.0851    | 0.0000          | 0.0000    | -0.9029     |            |             |            |            |
| home_local_start_mismatch  | 0.0295    | 0.0000          | 0.0000    | 0.4734      |            |             |            |            |
| rest_diff                  | -0.0308   | 0.0000          | 0.0000    | 0.8055      |            |             |            |            |
| home_days_into_season_sqrt | -0.0099   | 0.0000          | 0.0000    | -0.2367     | -0.2226    | mean,disp   | -0.0967    | 0.0000     |
| dyn_var_diff               | -0.0365   | 0.0000          | 0.0000    | -3.0717     |            |             |            |            |
| away_games_played_prior    | -0.0001   | 0.0000          | 0.0000    | 0.1844      | 0.0065     | mean,disp   | 0.0037     |            |
| home_games_played_prior    | 0.0001    | 0.0000          | 0.0000    | 0.1370      | 0.0092     | mean,disp   | 0.0043     | 0.0000     |
| home_days_into_season      | -0.0002   | 0.0000          | 0.0000    | -0.2681     |            |             | -0.0020    |            |

### Coefficient Detail

### GLM Ridge (`glm_ridge`)

Comparison fit with original-scale linear coefficients. Current comparison params: c=0.015625; l1_ratio=None; lambda=64; model_name=glm_ridge; penalty_family=ridge. Fit on the latest comparison fit window before the final holdout.

| Term                       | Base Feature               | Role   | Beta    | Scale            | Note |
| -------------------------- | -------------------------- | ------ | ------- | ---------------- | ---- |
| home_starter_runs_allowed  | home_starter_runs_allowed  | linear | -0.5548 | original_feature |      |
| away_starter_runs_allowed  | away_starter_runs_allowed  | linear | 0.5212  | original_feature |      |
| elo_home_prob              | elo_home_prob              | linear | -0.0146 | original_feature |      |
| elo_home_pre               | elo_home_pre               | linear | 0.0004  | original_feature |      |
| dyn_home_prob              | dyn_home_prob              | linear | 0.2482  | original_feature |      |
| home_ewm_run_diff          | home_ewm_run_diff          | linear | 0.0114  | original_feature |      |
| home_r14_run_diff          | home_r14_run_diff          | linear | 0.0077  | original_feature |      |
| away_r5_runs_for           | away_r5_runs_for           | linear | -0.0532 | original_feature |      |
| dyn_home_mean              | dyn_home_mean              | linear | 0.1081  | original_feature |      |
| elo_home_prob_hinge_052    | elo_home_prob_hinge_052    | linear | -0.1236 | original_feature |      |
| away_r5_run_diff           | away_r5_run_diff           | linear | -0.0191 | original_feature |      |
| away_ewm_run_diff          | away_ewm_run_diff          | linear | 0.0014  | original_feature |      |
| home_r5_run_diff           | home_r5_run_diff           | linear | -0.0079 | original_feature |      |
| diff_form_win_rate         | diff_form_win_rate         | linear | -0.0817 | original_feature |      |
| home_win_rate_ewm          | home_win_rate_ewm          | linear | 0.1062  | original_feature |      |
| home_r14_runs_against      | home_r14_runs_against      | linear | -0.0079 | original_feature |      |
| home_ewm_runs_against      | home_ewm_runs_against      | linear | 0.0113  | original_feature |      |
| dyn_away_mean              | dyn_away_mean              | linear | -0.0478 | original_feature |      |
| away_r14_run_diff          | away_r14_run_diff          | linear | -0.0017 | original_feature |      |
| home_ewm_runs_for          | home_ewm_runs_for          | linear | 0.0374  | original_feature |      |
| away_ewm_runs_for          | away_ewm_runs_for          | linear | 0.0292  | original_feature |      |
| home_r14_runs_for          | home_r14_runs_for          | linear | 0.0072  | original_feature |      |
| home_r5_runs_against       | home_r5_runs_against       | linear | 0.0173  | original_feature |      |
| elo_away_pre               | elo_away_pre               | linear | 0.0003  | original_feature |      |
| away_r14_runs_against      | away_r14_runs_against      | linear | 0.0114  | original_feature |      |
| away_win_rate_ewm          | away_win_rate_ewm          | linear | 0.3351  | original_feature |      |
| away_ewm_runs_against      | away_ewm_runs_against      | linear | 0.0269  | original_feature |      |
| home_r5_runs_for           | home_r5_runs_for           | linear | 0.0023  | original_feature |      |
| away_r14_runs_for          | away_r14_runs_for          | linear | 0.0083  | original_feature |      |
| away_3in4                  | away_3in4                  | linear | -0.1202 | original_feature |      |
| home_tz_change             | home_tz_change             | linear | -0.0529 | original_feature |      |
| away_r5_runs_against       | away_r5_runs_against       | linear | -0.0148 | original_feature |      |
| home_3in4                  | home_3in4                  | linear | 0.0201  | original_feature |      |
| away_travel_miles          | away_travel_miles          | linear | -0.0001 | original_feature |      |
| home_4in6                  | home_4in6                  | linear | -0.0083 | original_feature |      |
| home_rest_days_x           | home_rest_days_x           | linear | 0.0374  | original_feature |      |
| away_rest_days_x           | away_rest_days_x           | linear | 0.0420  | original_feature |      |
| home_travel_miles          | home_travel_miles          | linear | -0.0001 | original_feature |      |
| away_tz_change             | away_tz_change             | linear | -0.0190 | original_feature |      |
| away_b2b_x                 | away_b2b_x                 | linear | 0.0329  | original_feature |      |
| home_b2b_x                 | home_b2b_x                 | linear | -0.0013 | original_feature |      |
| travel_diff                | travel_diff                | linear | 0.0000  | original_feature |      |
| away_4in6                  | away_4in6                  | linear | 0.0851  | original_feature |      |
| home_local_start_mismatch  | home_local_start_mismatch  | linear | 0.0295  | original_feature |      |
| rest_diff                  | rest_diff                  | linear | -0.0308 | original_feature |      |
| home_days_into_season_sqrt | home_days_into_season_sqrt | linear | -0.0099 | original_feature |      |
| dyn_var_diff               | dyn_var_diff               | linear | -0.0365 | original_feature |      |
| away_games_played_prior    | away_games_played_prior    | linear | -0.0001 | original_feature |      |
| home_games_played_prior    | home_games_played_prior    | linear | 0.0001  | original_feature |      |
| home_days_into_season      | home_days_into_season      | linear | -0.0002 | original_feature |      |

### Elastic Net GLM (`glm_elastic_net`)

Comparison fit with original-scale linear coefficients. Current comparison params: c=0.03125; l1_ratio=0.85; lambda=32; model_name=glm_elastic_net; penalty_family=elastic_net. Fit on the latest comparison fit window before the final holdout.

| Term                       | Base Feature               | Role   | Beta    | Scale            | Note |
| -------------------------- | -------------------------- | ------ | ------- | ---------------- | ---- |
| home_starter_runs_allowed  | home_starter_runs_allowed  | linear | -1.0651 | original_feature |      |
| away_starter_runs_allowed  | away_starter_runs_allowed  | linear | 1.0287  | original_feature |      |
| elo_home_prob              | elo_home_prob              | linear | 0.0000  | original_feature |      |
| elo_home_pre               | elo_home_pre               | linear | 0.0000  | original_feature |      |
| dyn_home_prob              | dyn_home_prob              | linear | 0.0000  | original_feature |      |
| home_ewm_run_diff          | home_ewm_run_diff          | linear | 0.0000  | original_feature |      |
| home_r14_run_diff          | home_r14_run_diff          | linear | 0.0000  | original_feature |      |
| away_r5_runs_for           | away_r5_runs_for           | linear | 0.0000  | original_feature |      |
| dyn_home_mean              | dyn_home_mean              | linear | 0.0000  | original_feature |      |
| elo_home_prob_hinge_052    | elo_home_prob_hinge_052    | linear | 0.0000  | original_feature |      |
| away_r5_run_diff           | away_r5_run_diff           | linear | 0.0000  | original_feature |      |
| away_ewm_run_diff          | away_ewm_run_diff          | linear | 0.0000  | original_feature |      |
| home_r5_run_diff           | home_r5_run_diff           | linear | 0.0000  | original_feature |      |
| diff_form_win_rate         | diff_form_win_rate         | linear | 0.0000  | original_feature |      |
| home_win_rate_ewm          | home_win_rate_ewm          | linear | 0.0000  | original_feature |      |
| home_r14_runs_against      | home_r14_runs_against      | linear | 0.0000  | original_feature |      |
| home_ewm_runs_against      | home_ewm_runs_against      | linear | 0.0000  | original_feature |      |
| dyn_away_mean              | dyn_away_mean              | linear | 0.0000  | original_feature |      |
| away_r14_run_diff          | away_r14_run_diff          | linear | 0.0000  | original_feature |      |
| home_ewm_runs_for          | home_ewm_runs_for          | linear | 0.0000  | original_feature |      |
| away_ewm_runs_for          | away_ewm_runs_for          | linear | 0.0000  | original_feature |      |
| home_r14_runs_for          | home_r14_runs_for          | linear | 0.0000  | original_feature |      |
| home_r5_runs_against       | home_r5_runs_against       | linear | 0.0000  | original_feature |      |
| elo_away_pre               | elo_away_pre               | linear | 0.0000  | original_feature |      |
| away_r14_runs_against      | away_r14_runs_against      | linear | 0.0000  | original_feature |      |
| away_win_rate_ewm          | away_win_rate_ewm          | linear | 0.0000  | original_feature |      |
| away_ewm_runs_against      | away_ewm_runs_against      | linear | 0.0000  | original_feature |      |
| home_r5_runs_for           | home_r5_runs_for           | linear | 0.0000  | original_feature |      |
| away_r14_runs_for          | away_r14_runs_for          | linear | 0.0000  | original_feature |      |
| away_3in4                  | away_3in4                  | linear | 0.0000  | original_feature |      |
| home_tz_change             | home_tz_change             | linear | 0.0000  | original_feature |      |
| away_r5_runs_against       | away_r5_runs_against       | linear | 0.0000  | original_feature |      |
| home_3in4                  | home_3in4                  | linear | 0.0000  | original_feature |      |
| away_travel_miles          | away_travel_miles          | linear | 0.0000  | original_feature |      |
| home_4in6                  | home_4in6                  | linear | 0.0000  | original_feature |      |
| home_rest_days_x           | home_rest_days_x           | linear | 0.0000  | original_feature |      |
| away_rest_days_x           | away_rest_days_x           | linear | 0.0000  | original_feature |      |
| home_travel_miles          | home_travel_miles          | linear | 0.0000  | original_feature |      |
| away_tz_change             | away_tz_change             | linear | 0.0000  | original_feature |      |
| away_b2b_x                 | away_b2b_x                 | linear | 0.0000  | original_feature |      |
| home_b2b_x                 | home_b2b_x                 | linear | 0.0000  | original_feature |      |
| travel_diff                | travel_diff                | linear | 0.0000  | original_feature |      |
| away_4in6                  | away_4in6                  | linear | 0.0000  | original_feature |      |
| home_local_start_mismatch  | home_local_start_mismatch  | linear | 0.0000  | original_feature |      |
| rest_diff                  | rest_diff                  | linear | 0.0000  | original_feature |      |
| home_days_into_season_sqrt | home_days_into_season_sqrt | linear | 0.0000  | original_feature |      |
| dyn_var_diff               | dyn_var_diff               | linear | 0.0000  | original_feature |      |
| away_games_played_prior    | away_games_played_prior    | linear | 0.0000  | original_feature |      |
| home_games_played_prior    | home_games_played_prior    | linear | 0.0000  | original_feature |      |
| home_days_into_season      | home_days_into_season      | linear | 0.0000  | original_feature |      |

### Lasso GLM (`glm_lasso`)

Comparison fit with original-scale linear coefficients. Current comparison params: c=0.0416667; l1_ratio=None; lambda=24; model_name=glm_lasso; penalty_family=lasso. Fit on the latest comparison fit window before the final holdout.

| Term                       | Base Feature               | Role   | Beta    | Scale            | Note |
| -------------------------- | -------------------------- | ------ | ------- | ---------------- | ---- |
| home_starter_runs_allowed  | home_starter_runs_allowed  | linear | -1.4934 | original_feature |      |
| away_starter_runs_allowed  | away_starter_runs_allowed  | linear | 1.4623  | original_feature |      |
| elo_home_prob              | elo_home_prob              | linear | 0.0000  | original_feature |      |
| elo_home_pre               | elo_home_pre               | linear | 0.0000  | original_feature |      |
| dyn_home_prob              | dyn_home_prob              | linear | 0.0000  | original_feature |      |
| home_ewm_run_diff          | home_ewm_run_diff          | linear | 0.0000  | original_feature |      |
| home_r14_run_diff          | home_r14_run_diff          | linear | 0.0000  | original_feature |      |
| away_r5_runs_for           | away_r5_runs_for           | linear | 0.0000  | original_feature |      |
| dyn_home_mean              | dyn_home_mean              | linear | 0.0000  | original_feature |      |
| elo_home_prob_hinge_052    | elo_home_prob_hinge_052    | linear | 0.0000  | original_feature |      |
| away_r5_run_diff           | away_r5_run_diff           | linear | 0.0000  | original_feature |      |
| away_ewm_run_diff          | away_ewm_run_diff          | linear | 0.0000  | original_feature |      |
| home_r5_run_diff           | home_r5_run_diff           | linear | 0.0000  | original_feature |      |
| diff_form_win_rate         | diff_form_win_rate         | linear | 0.0000  | original_feature |      |
| home_win_rate_ewm          | home_win_rate_ewm          | linear | 0.0000  | original_feature |      |
| home_r14_runs_against      | home_r14_runs_against      | linear | 0.0000  | original_feature |      |
| home_ewm_runs_against      | home_ewm_runs_against      | linear | 0.0000  | original_feature |      |
| dyn_away_mean              | dyn_away_mean              | linear | 0.0000  | original_feature |      |
| away_r14_run_diff          | away_r14_run_diff          | linear | 0.0000  | original_feature |      |
| home_ewm_runs_for          | home_ewm_runs_for          | linear | 0.0000  | original_feature |      |
| away_ewm_runs_for          | away_ewm_runs_for          | linear | 0.0000  | original_feature |      |
| home_r14_runs_for          | home_r14_runs_for          | linear | 0.0000  | original_feature |      |
| home_r5_runs_against       | home_r5_runs_against       | linear | 0.0000  | original_feature |      |
| elo_away_pre               | elo_away_pre               | linear | 0.0000  | original_feature |      |
| away_r14_runs_against      | away_r14_runs_against      | linear | 0.0000  | original_feature |      |
| away_win_rate_ewm          | away_win_rate_ewm          | linear | 0.0000  | original_feature |      |
| away_ewm_runs_against      | away_ewm_runs_against      | linear | 0.0000  | original_feature |      |
| home_r5_runs_for           | home_r5_runs_for           | linear | 0.0000  | original_feature |      |
| away_r14_runs_for          | away_r14_runs_for          | linear | 0.0000  | original_feature |      |
| away_3in4                  | away_3in4                  | linear | 0.0000  | original_feature |      |
| home_tz_change             | home_tz_change             | linear | 0.0000  | original_feature |      |
| away_r5_runs_against       | away_r5_runs_against       | linear | 0.0000  | original_feature |      |
| home_3in4                  | home_3in4                  | linear | 0.0000  | original_feature |      |
| away_travel_miles          | away_travel_miles          | linear | 0.0000  | original_feature |      |
| home_4in6                  | home_4in6                  | linear | 0.0000  | original_feature |      |
| home_rest_days_x           | home_rest_days_x           | linear | 0.0000  | original_feature |      |
| away_rest_days_x           | away_rest_days_x           | linear | 0.0000  | original_feature |      |
| home_travel_miles          | home_travel_miles          | linear | 0.0000  | original_feature |      |
| away_tz_change             | away_tz_change             | linear | 0.0000  | original_feature |      |
| away_b2b_x                 | away_b2b_x                 | linear | 0.0000  | original_feature |      |
| home_b2b_x                 | home_b2b_x                 | linear | 0.0000  | original_feature |      |
| travel_diff                | travel_diff                | linear | 0.0000  | original_feature |      |
| away_4in6                  | away_4in6                  | linear | 0.0000  | original_feature |      |
| home_local_start_mismatch  | home_local_start_mismatch  | linear | 0.0000  | original_feature |      |
| rest_diff                  | rest_diff                  | linear | 0.0000  | original_feature |      |
| home_days_into_season_sqrt | home_days_into_season_sqrt | linear | 0.0000  | original_feature |      |
| dyn_var_diff               | dyn_var_diff               | linear | 0.0000  | original_feature |      |
| away_games_played_prior    | away_games_played_prior    | linear | 0.0000  | original_feature |      |
| home_games_played_prior    | home_games_played_prior    | linear | 0.0000  | original_feature |      |
| home_days_into_season      | home_days_into_season      | linear | 0.0000  | original_feature |      |

### Vanilla GLM (`glm_vanilla`)

Comparison fit with original-scale GLM coefficients. Current comparison params: {}. Fit on the latest comparison fit window before the final holdout.

| Term                       | Base Feature               | Role   | Beta     | Scale            | Note |
| -------------------------- | -------------------------- | ------ | -------- | ---------------- | ---- |
| home_starter_runs_allowed  | home_starter_runs_allowed  | linear | -27.0669 | original_feature |      |
| away_starter_runs_allowed  | away_starter_runs_allowed  | linear | 27.0951  | original_feature |      |
| elo_home_prob              | elo_home_prob              | linear | 28.9304  | original_feature |      |
| elo_home_pre               | elo_home_pre               | linear | -0.0200  | original_feature |      |
| dyn_home_prob              | dyn_home_prob              | linear | 2.4218   | original_feature |      |
| home_ewm_run_diff          | home_ewm_run_diff          | linear | -1.2911  | original_feature |      |
| home_r14_run_diff          | home_r14_run_diff          | linear | 0.0930   | original_feature |      |
| away_r5_runs_for           | away_r5_runs_for           | linear | -0.0447  | original_feature |      |
| dyn_home_mean              | dyn_home_mean              | linear | 0.5404   | original_feature |      |
| elo_home_prob_hinge_052    | elo_home_prob_hinge_052    | linear | -0.6180  | original_feature |      |
| away_r5_run_diff           | away_r5_run_diff           | linear | -0.0658  | original_feature |      |
| away_ewm_run_diff          | away_ewm_run_diff          | linear | 11.6217  | original_feature |      |
| home_r5_run_diff           | home_r5_run_diff           | linear | -0.0576  | original_feature |      |
| diff_form_win_rate         | diff_form_win_rate         | linear | 10.0700  | original_feature |      |
| home_win_rate_ewm          | home_win_rate_ewm          | linear | -12.4059 | original_feature |      |
| home_r14_runs_against      | home_r14_runs_against      | linear | -0.2045  | original_feature |      |
| home_ewm_runs_against      | home_ewm_runs_against      | linear | -1.2307  | original_feature |      |
| dyn_away_mean              | dyn_away_mean              | linear | -0.1494  | original_feature |      |
| away_r14_run_diff          | away_r14_run_diff          | linear | 29.6652  | original_feature |      |
| home_ewm_runs_for          | home_ewm_runs_for          | linear | 1.4804   | original_feature |      |
| away_ewm_runs_for          | away_ewm_runs_for          | linear | -11.7112 | original_feature |      |
| home_r14_runs_for          | home_r14_runs_for          | linear | -0.0454  | original_feature |      |
| home_r5_runs_against       | home_r5_runs_against       | linear | 0.1108   | original_feature |      |
| elo_away_pre               | elo_away_pre               | linear | 0.0178   | original_feature |      |
| away_r14_runs_against      | away_r14_runs_against      | linear | 29.5393  | original_feature |      |
| away_win_rate_ewm          | away_win_rate_ewm          | linear | 11.9795  | original_feature |      |
| away_ewm_runs_against      | away_ewm_runs_against      | linear | 11.5555  | original_feature |      |
| home_r5_runs_for           | home_r5_runs_for           | linear | 0.0010   | original_feature |      |
| away_r14_runs_for          | away_r14_runs_for          | linear | -29.3684 | original_feature |      |
| away_3in4                  | away_3in4                  | linear | -1.4604  | original_feature |      |
| home_tz_change             | home_tz_change             | linear | -0.1516  | original_feature |      |
| away_r5_runs_against       | away_r5_runs_against       | linear | 0.0906   | original_feature |      |
| home_3in4                  | home_3in4                  | linear | 1.6311   | original_feature |      |
| away_travel_miles          | away_travel_miles          | linear | 0.0002   | original_feature |      |
| home_4in6                  | home_4in6                  | linear | -1.5739  | original_feature |      |
| home_rest_days_x           | home_rest_days_x           | linear | 0.1780   | original_feature |      |
| away_rest_days_x           | away_rest_days_x           | linear | 0.0789   | original_feature |      |
| home_travel_miles          | home_travel_miles          | linear | -0.0001  | original_feature |      |
| away_tz_change             | away_tz_change             | linear | 0.2538   | original_feature |      |
| away_b2b_x                 | away_b2b_x                 | linear | 0.4614   | original_feature |      |
| home_b2b_x                 | home_b2b_x                 | linear | -0.2457  | original_feature |      |
| travel_diff                | travel_diff                | linear | -0.0002  | original_feature |      |
| away_4in6                  | away_4in6                  | linear | -0.9029  | original_feature |      |
| home_local_start_mismatch  | home_local_start_mismatch  | linear | 0.4734   | original_feature |      |
| rest_diff                  | rest_diff                  | linear | 0.8055   | original_feature |      |
| home_days_into_season_sqrt | home_days_into_season_sqrt | linear | -0.2367  | original_feature |      |
| dyn_var_diff               | dyn_var_diff               | linear | -3.0717  | original_feature |      |
| away_games_played_prior    | away_games_played_prior    | linear | 0.1844   | original_feature |      |
| home_games_played_prior    | home_games_played_prior    | linear | 0.1370   | original_feature |      |
| home_days_into_season      | home_days_into_season      | linear | -0.2681  | original_feature |      |

### GLMM Logit (`glmm_logit`)

Fixed-effect coefficients only; team random-effect terms are omitted from the matrix. Current comparison params: feature_cap=6. Fit on the latest comparison fit window before the final holdout.

| Term                       | Base Feature               | Role         | Beta    | Scale            | Note |
| -------------------------- | -------------------------- | ------------ | ------- | ---------------- | ---- |
| home_starter_runs_allowed  | home_starter_runs_allowed  | fixed_effect | -3.4840 | original_feature |      |
| away_starter_runs_allowed  | away_starter_runs_allowed  | fixed_effect | 3.4632  | original_feature |      |
| home_days_into_season_sqrt | home_days_into_season_sqrt | fixed_effect | -0.2226 | original_feature |      |
| home_games_played_prior    | home_games_played_prior    | fixed_effect | 0.0092  | original_feature |      |
| away_games_played_prior    | away_games_played_prior    | fixed_effect | 0.0065  | original_feature |      |
| away_r5_runs_for           | away_r5_runs_for           | fixed_effect | -0.0473 | original_feature |      |

### DGLM Margin (`dglm_margin`)

Mean and dispersion coefficients are both shown for each selected feature. Current comparison params: feature_cap=6; iterations=1. Fit on the latest comparison fit window before the final holdout.

| Term                                   | Base Feature               | Role       | Beta    | Scale            | Note |
| -------------------------------------- | -------------------------- | ---------- | ------- | ---------------- | ---- |
| mean::home_starter_runs_allowed        | home_starter_runs_allowed  | mean       | -1.0000 | original_feature |      |
| dispersion::home_starter_runs_allowed  | home_starter_runs_allowed  | dispersion | 0.0000  | original_feature |      |
| mean::away_starter_runs_allowed        | away_starter_runs_allowed  | mean       | 1.0000  | original_feature |      |
| dispersion::away_starter_runs_allowed  | away_starter_runs_allowed  | dispersion | 0.0000  | original_feature |      |
| mean::home_days_into_season_sqrt       | home_days_into_season_sqrt | mean       | 0.0000  | original_feature |      |
| dispersion::home_days_into_season_sqrt | home_days_into_season_sqrt | dispersion | -0.0000 | original_feature |      |
| mean::home_games_played_prior          | home_games_played_prior    | mean       | -0.0000 | original_feature |      |
| dispersion::home_games_played_prior    | home_games_played_prior    | dispersion | 0.0000  | original_feature |      |
| mean::away_games_played_prior          | away_games_played_prior    | mean       | -0.0000 | original_feature |      |
| dispersion::away_games_played_prior    | away_games_played_prior    | dispersion | 0.0000  | original_feature |      |
| mean::away_r5_runs_for                 | away_r5_runs_for           | mean       | -0.0000 | original_feature |      |
| dispersion::away_r5_runs_for           | away_r5_runs_for           | dispersion | -0.0000 | original_feature |      |

### GAM Spline (`gam_spline`)

Spline rows are basis-term coefficients, not a single beta on the raw predictor scale. Current comparison params: c=0.5; feature_cap=3; n_knots=5. Fit on the latest comparison fit window before the final holdout.

| Term                                 | Base Feature               | Role         | Beta     | Scale            | Note |
| ------------------------------------ | -------------------------- | ------------ | -------- | ---------------- | ---- |
| home_days_into_season_sqrt           | home_days_into_season_sqrt | linear       | -0.0967  | original_feature |      |
| home_games_played_prior              | home_games_played_prior    | linear       | 0.0043   | original_feature |      |
| away_games_played_prior              | away_games_played_prior    | linear       | 0.0037   | original_feature |      |
| away_r5_runs_for                     | away_r5_runs_for           | linear       | -0.0384  | original_feature |      |
| home_days_into_season                | home_days_into_season      | linear       | -0.0020  | original_feature |      |
| home_starter_runs_allowed::spline_01 | home_starter_runs_allowed  | spline_basis | 27.0063  | basis_term       |      |
| home_starter_runs_allowed::spline_02 | home_starter_runs_allowed  | spline_basis | 11.3008  | basis_term       |      |
| home_starter_runs_allowed::spline_03 | home_starter_runs_allowed  | spline_basis | 6.6722   | basis_term       |      |
| home_starter_runs_allowed::spline_04 | home_starter_runs_allowed  | spline_basis | -1.4358  | basis_term       |      |
| home_starter_runs_allowed::spline_05 | home_starter_runs_allowed  | spline_basis | -16.0678 | basis_term       |      |
| home_starter_runs_allowed::spline_06 | home_starter_runs_allowed  | spline_basis | -25.8729 | basis_term       |      |
| away_starter_runs_allowed::spline_01 | away_starter_runs_allowed  | spline_basis | -28.4004 | basis_term       |      |
| away_starter_runs_allowed::spline_02 | away_starter_runs_allowed  | spline_basis | -10.9659 | basis_term       |      |
| away_starter_runs_allowed::spline_03 | away_starter_runs_allowed  | spline_basis | -6.7180  | basis_term       |      |
| away_starter_runs_allowed::spline_04 | away_starter_runs_allowed  | spline_basis | 2.0387   | basis_term       |      |
| away_starter_runs_allowed::spline_05 | away_starter_runs_allowed  | spline_basis | 15.4196  | basis_term       |      |
| away_starter_runs_allowed::spline_06 | away_starter_runs_allowed  | spline_basis | 29.9695  | basis_term       |      |
| dyn_home_mean::spline_01             | dyn_home_mean              | spline_basis | 5.0393   | basis_term       |      |
| dyn_home_mean::spline_02             | dyn_home_mean              | spline_basis | -0.6575  | basis_term       |      |
| dyn_home_mean::spline_03             | dyn_home_mean              | spline_basis | -0.2269  | basis_term       |      |
| dyn_home_mean::spline_04             | dyn_home_mean              | spline_basis | -0.0868  | basis_term       |      |
| dyn_home_mean::spline_05             | dyn_home_mean              | spline_basis | 0.3879   | basis_term       |      |
| dyn_home_mean::spline_06             | dyn_home_mean              | spline_basis | -0.2145  | basis_term       |      |

### MARS Hinge (`mars_hinge`)

Hinge rows are basis-term coefficients on the MARS proxy basis. Current comparison params: c=0.1; feature_cap=2; interaction_degree=1; knots_per_feature=3. Fit on the latest comparison fit window before the final holdout.

| Term                                   | Base Feature               | Role        | Beta    | Scale            | Note |
| -------------------------------------- | -------------------------- | ----------- | ------- | ---------------- | ---- |
| home_starter_runs_allowed              | home_starter_runs_allowed  | linear      | -2.2566 | original_feature |      |
| away_starter_runs_allowed              | away_starter_runs_allowed  | linear      | 2.2330  | original_feature |      |
| home_days_into_season_sqrt             | home_days_into_season_sqrt | linear      | 0.0000  | original_feature |      |
| home_games_played_prior                | home_games_played_prior    | linear      | 0.0000  | original_feature |      |
| diff_form_win_rate_hinge_pos_-0.348567 | diff_form_win_rate         | hinge_basis | 0.0000  | basis_term       |      |
| diff_form_win_rate_hinge_neg_-0.348567 | diff_form_win_rate         | hinge_basis | 0.0000  | basis_term       |      |
| diff_form_win_rate_hinge_pos_0.005753  | diff_form_win_rate         | hinge_basis | 0.0000  | basis_term       |      |
| diff_form_win_rate_hinge_neg_0.005753  | diff_form_win_rate         | hinge_basis | 0.0000  | basis_term       |      |
| diff_form_win_rate_hinge_pos_0.353978  | diff_form_win_rate         | hinge_basis | 0.0000  | basis_term       |      |
| diff_form_win_rate_hinge_neg_0.353978  | diff_form_win_rate         | hinge_basis | 0.0000  | basis_term       |      |

## Notes

- Linear production coefficients use original-feature scale when available.
- GAM spline rows and MARS hinge rows are basis-term coefficients, not a single raw-feature beta.
- GLMM rows include fixed effects only in the matrix; team random effects are omitted.
- DGLM rows include both mean and dispersion components for each selected feature.
