NBA all-teams next-game report as of 2026-03-04T14:44:02+00:00.

| Home Team | Away Team | Date | ensemble | elo_baseline | glm_logit | dynamic_rating | rf | goals_poisson | gbdt | two_stage | bayes_bt_state_space | bayes_goals | simulation_first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAC | IND | 2026-03-05 | 79.7% | 70.4% | 82.9% | 89.0% | 82.4% | 71.1% | 94.7% | 61.6% | 79.2% | 65.3% | 81.9% |
| ORL | DAL | 2026-03-06 | 74.6% | 64.5% | 71.6% | 86.4% | 81.1% | 61.3% | 94.2% | 53.4% | 72.9% | 70.7% | 76.7% |
| PHX | CHI | 2026-03-06 | 68.9% | 66.5% | 81.3% | 80.0% | 76.8% | 62.9% | 93.6% | 58.2% | 79.4% | 54.2% | 46.4% |
| HOU | GSW | 2026-03-06 | 68.9% | 61.8% | 72.8% | 79.7% | 71.9% | 62.7% | 71.5% | 39.8% | 71.4% | 65.8% | 71.1% |
| PHI | UTA | 2026-03-05 | 66.5% | 68.6% | 70.2% | 85.8% | 61.3% | 67.8% | 80.1% | 52.3% | 62.4% | 57.9% | 47.6% |
| BOS | CHA | 2026-03-05 | 59.3% | 61.7% | 69.2% | 57.1% | 57.4% | 63.3% | 73.5% | 34.2% | 65.0% | 60.6% | 42.9% |
| CLE | BOS | 2026-03-08 | 52.1% | 48.3% | 46.5% | 53.6% | 48.2% | 42.7% | 73.7% | 19.7% | 51.1% | 73.0% | 64.4% |
| MIN | GSW | 2026-01-24 | 51.8% | 53.1% | 65.7% | 39.9% | 48.1% | 61.4% | 18.9% | 35.2% | 63.4% | 72.6% | 51.6% |
| NYK | OKC | 2026-03-05 | 47.9% | 46.7% | 37.5% | 61.2% | 54.2% | 45.9% | 68.0% | 17.2% | 40.7% | 60.7% | 52.5% |
| SAS | DET | 2026-03-06 | 46.9% | 53.1% | 36.4% | 57.0% | 56.6% | 53.7% | 14.0% | 17.3% | 39.5% | 69.8% | 56.3% |
| WAS | UTA | 2026-03-06 | 44.7% | 52.0% | 48.8% | 64.7% | 36.6% | 47.5% | 46.0% | 38.5% | 39.5% | 48.4% | 42.0% |
| MIL | DAL | 2026-01-26 | 42.5% | 51.1% | 48.8% | 38.8% | 41.1% | 51.7% | 45.0% | 35.6% | 47.4% | 60.1% | 28.7% |
| CHI | MIA | 2026-01-09 | 40.6% | 49.8% | 39.6% | 39.7% | 52.1% | 40.9% | 27.2% | 52.5% | 39.9% | 56.2% | 40.0% |
| SAC | NOP | 2026-03-06 | 36.5% | 43.8% | 51.9% | 26.8% | 28.2% | 39.0% | 49.2% | 27.0% | 32.0% | 51.9% | 37.5% |
| MEM | DEN | 2026-01-25 | 31.9% | 37.6% | 34.5% | 23.8% | 39.6% | 39.2% | 12.7% | 26.5% | 34.5% | 55.7% | 39.6% |

Model trust guide (super brief): each line is Built on / Good at / Watch out.
- `ensemble`: All models combined. Best default pick. Can share the same blind spot.
- `elo_baseline`: Standard sports betting baseline based on past wins/losses. Good long-run read. Slow on sudden changes.
- `glm_logit`: Statistical model that uses a checklist. Usually steady. Weird matchups can slip through.
- `dynamic_rating`: Hot/cold meter. Good for momentum. Can overreact to short streaks.
- `rf`: Machine learning model that blends many different predictions from random slices of past games. Good at smoothing out flukes. Can be too cautious on close matchups.
- `goals_poisson`: Score-based model. Good for normal scoring games. Messy games hurt it.
- `gbdt`: Machine learning model that finds hidden combos. Sometimes too confident.
- `two_stage`: Machine learning model with two steps: first predicts game type (fast/slow, close/lopsided), then predicts winner. Good when style matchups matter. If step 1 is wrong, final pick can be wrong.
- `bayes_bt_state_space`: Tracks team strength after every game and gives a range, not just one number. Good for spotting rising/falling teams with uncertainty shown. Can move fast after injuries, trades, or short weird stretches.
- `bayes_goals`: Scoring strength + confidence meter. Good trend read. Can lag sudden lineup changes.
- `simulation_first`: Runs the matchup thousands of times using set assumptions (team strength, pace, and scoring). Good for seeing different paths. If those assumptions are off, this number can be off.
