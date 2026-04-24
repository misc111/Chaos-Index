# Baseball Models Research

## Purpose

This document is a research map for the MLB-first rebuild. The product goal is not to chase arbitrary machine-learning novelty. It is to use the repo's actuarial model lane, vanilla GLMs, ridge/lasso/elastic net, market-offset and prior-offset lasso credibility, GLMM, DGLM, GAM, MARS-style hinge discovery, and theory-compatible ensembles, to produce pregame MLB predictions for moneyline, runline, and totals.

The useful baseball literature points in the same direction as the actuarial contract: decompose the game into durable rate processes, shrink noisy observations aggressively, model context explicitly, validate with rolling point-in-time evidence, and treat the market as a strong benchmark or offset rather than as the model itself.

## Executive Takeaways

1. **The durable spine is weighted history plus regression.** Marcel, ZiPS, PECOTA, Steamer, and modern public projection practice all converge on recent multi-season performance, age adjustment, playing-time estimation, and regression toward an appropriate mean. The exact proprietary machinery varies, but the repeatable pattern is actuarial credibility, not black-box magic.
2. **Build runs first, then translate to wins and markets.** Pythagorean expectation, Markov run models, and run-distribution work all say that run scoring/run prevention is the more natural modeling layer. Moneyline is downstream from expected run distribution. Runline and totals need the joint score distribution, not just a binary win probability.
3. **Prefer component skills over outcomes.** K%, BB%, HR, HBP, batted-ball quality, platoon splits, park, defense, bullpen availability, and weather carry cleaner signal than one-season ERA, win-loss record, RBI, saves, or raw recent streaks.
4. **Use Statcast as signal, not prophecy.** Exit velocity, launch angle, barrels, hard-hit rate, pitch velocity/movement/spin/extension, xwOBA/xBA/xSLG, OAA, and catcher framing are useful pregame inputs. They still need lagging, shrinkage, and timestamp discipline. MLB itself frames xERA/xwOBA as contact-quality/context diagnostics, not automatic next-game predictions.
5. **Market odds belong in an explicit lane.** Vig-free open/close probabilities are powerful priors and benchmarks. The repo should implement market offsets and complement models so the structural baseball model can learn residual edge without becoming a market copier.

## The Giants To Stand On

### Marcel And The Projection Baseline

Tom Tango's Marcel is valuable because it is deliberately plain: three years of MLB data, heavier weight on recent seasons, regression to league mean, and age adjustment. It is the minimum competent projection system, which makes it the correct baseline to beat. The important lesson is not the exact coefficients. The important lesson is that a simple credibility-weighted baseline is hard to crush.

Repo mapping:

- Direct theory implementation: credibility-weighted priors for players, teams, pitchers, bullpens, and defensive groups.
- GLM feature inputs: lagged multi-season rates, recency weights, exposure/reliability weights, age terms, and shrinkage-to-mean indicators.
- Governance rule: no complex challenger should be promoted unless it beats this simple regressed baseline on rolling backtests.

### DIPS, FIP, And Pitcher Component Thinking

Defense Independent Pitching Statistics and FIP changed pitcher evaluation by separating outcomes a pitcher controls most directly, strikeouts, walks, hit batters, and home runs, from balls in play and team defense. MLB's glossary defines FIP around those pitcher-controlled events and MLB's BABIP glossary explicitly warns that high or low BABIP usually regresses, while still allowing for skill through weak or hard contact.

Repo mapping:

- Direct theory implementation: pitcher run-prevention GLMs should include K%, BB%, HBP%, HR/PA, GB/FB mix, contact quality allowed, and workload/rest.
- Theory-compatible engineering support: split pitcher skill from defense/park/weather residuals.
- Avoid: using raw ERA, pitcher wins, or recent win/loss streaks as primary pitcher talent features.

### wOBA And Linear Run Values

wOBA is the long-lived offensive framework because it values batting events by run expectancy rather than counting every time on base equally. That makes it a natural bridge from baseball event data into actuarial severity/frequency thinking: events have different expected run values, and seasonal run environments move those weights.

Repo mapping:

- Direct theory implementation: offensive GLMs should prefer run-value or wOBA-family features over batting average, RBI, or OPS alone.
- Feature blocks: team and lineup projected wOBA/wRC-style skill, platoon-adjusted wOBA, ISO/power, K/BB discipline, baserunning value, and rolling quality-of-contact.
- Betting overlay: convert offensive and run-prevention outputs into expected runs, then into moneyline/runline/totals probabilities.

### Pythagorean Expectation And Run-To-Win Translation

Bill James' Pythagorean expectation remains a durable season-level bridge from runs scored and allowed to winning percentage. Miller derived a Weibull-based justification for the familiar exponent near 1.8, while later work showed more flexible forms can improve fit. Baseball Prospectus testing also emphasized that the run environment matters, and that a fixed exponent is a simplification.

Repo mapping:

- Direct theory implementation: use Pythagorean-style transformations as calibration checks for team expected runs and season-level latent team strength.
- GLM output layer: moneyline is best treated as a logit or paired-comparison probability after expected run distribution is estimated.
- Validation: compare model-implied team strength against Pythagorean/team-quality baselines.

### Markov Run Models

Bukiet, Harold, and Palacios' Markov-chain baseball model is still the canonical structural game engine: base-out states, batter abilities, run distributions by inning/game, batting order, and expected wins. For a pregame betting repo, a full Markov simulator may be too heavy for every model path, but it is an excellent generator of covariates and challenger distributions.

Repo mapping:

- Theory-compatible engineering support: use Markov outputs as pregame expected-runs covariates or benchmark simulations.
- GLMM/DGLM extension: pitcher, lineup, park, and bullpen states can enter as time-varying or random effects.
- Runline/totals: Markov or joint score models are more natural than deriving everything from a binary moneyline model.

### Empirical Bayes, Shrinkage, And Player True Talent

Baseball is a shrinkage sport. Jim Albert's baseball statistics work, including run expectancy and empirical-Bayes framing, sits near the same intellectual home as actuarial credibility: observed performance is a noisy sample from an underlying talent distribution. Short-sample BABIP, HR/FB, defensive metrics, reliever ERA, and clutch splits should be pulled hard toward priors.

Repo mapping:

- Direct theory implementation: prior-offset lasso credibility for player/team effects.
- GLMM: random player/team/park/catcher effects with partial pooling.
- Promotion evidence: show lift over no-shrinkage variants, not just headline accuracy.

## Durable Predictor Families

### Offense

Strong offensive predictors should be pregame, lagged, exposure-weighted, and park/platoon adjusted.

- Plate discipline: K%, BB%, chase/contact where available, and pitcher-hitter handedness interactions.
- Power/contact quality: ISO, HR/PA, hard-hit rate, barrel rate, exit velocity, launch-angle sweet spot, xSLG, and xwOBA.
- Overall run value: projected wOBA/wRC-style skill, lineup-weighted by expected starters.
- Baserunning: sprint speed, stolen-base value, extra-base-taking value, and catcher/pitcher running-game context.
- Playing time and lineup position: announced lineup, expected plate appearances, injuries, rest, and pinch-hit risk.

Actuarial fit:

- Logistic/binomial GLMs for event probabilities.
- Poisson/NB severity/frequency layers for runs.
- GAM/MARS for nonlinear launch-angle, exit-velocity, and weather interactions.
- Lasso/elastic net for high-dimensional platoon and lineup interactions.

### Starting Pitching

Durable pitcher modeling starts with components, not ERA.

- Defense-independent skills: K%, BB%, HBP%, HR/PA, ground-ball/fly-ball tendencies.
- Contact allowed: xwOBA allowed, barrel rate allowed, hard-hit rate allowed, EV/LA allowed, weak-contact indicators.
- Pitch traits: velocity, movement, spin, active spin, extension, pitch mix, pitch-type run values, and pitch-shape changes.
- Usage context: days rest, pitch counts, recent workload, injury/return status, opener/bulk role, expected innings.
- Handedness and matchup: pitcher hand against lineup handedness and platoon splits.

Actuarial fit:

- GLM/NB expected-runs model with pitcher component covariates.
- DGLM for changing pitcher talent, velocity changes, role changes, and injury-return uncertainty.
- GLMM for pitcher random effects, catcher effects, and team defense effects.

### Bullpen

Bullpen is a major pregame predictor because MLB games are increasingly bullpen-heavy.

- Talent: projected reliever K/BB/HR/contact quality, depth chart, leverage hierarchy.
- Availability: pitches thrown over the last 1, 2, and 3 days, back-to-back usage, extra-inning exposure, travel, and roster churn.
- Handedness: left/right reliever availability against opponent lineup pockets.
- Managerial pattern: closer/setup usage tendencies and opener/bulk setups.

Actuarial fit:

- Exposure-weighted bullpen run-prevention index.
- GLMM bullpen/team relief random effects.
- DGLM for short-term availability state, while keeping talent estimates shrunk.

### Defense, Catching, And Umpire Context

Defense matters, but it is noisy and often confounded with pitching and park.

- Defense: OAA, fielding run value, positional alignment, throwing/arm value where pregame lineups are known.
- Catching: framing runs, blocking, throwing/pop time, pitcher-catcher pair effects.
- Umpires: strike-zone tendencies and home bias can matter, but should be small, lagged, and only used when umpire assignment is known pregame.

Actuarial fit:

- GLMM random effects for catcher, park, defense group, and umpire.
- Ridge/lasso to prevent fragile defensive splits from dominating.
- Explicit label: this is theory-compatible engineering support, not a core betting edge until validated.

### Park, Weather, Roof, And Run Environment

Park and weather are not decoration; they are run-environment variables.

- Park: run factor, HR factor, handedness-specific park effects, outfield dimensions, altitude, foul territory, wall height.
- Weather: temperature, wind speed/direction, humidity, air pressure, precipitation risk.
- Roof: announced roof/open status and park-specific indoor/outdoor behavior.
- Ball environment: season/year effects, drag changes, and league run environment.

Alan Nathan's 2023 temperature analysis used Statcast and a GAM/logistic framework and found roughly a 1 percent increase in home runs per 1 degree Fahrenheit. Konaka's park-factor paper models plate appearances as batter team versus pitcher team plus ballpark, using logistic regression over more than 1.5 million MLB plate appearances.

Actuarial fit:

- GAM for smooth temperature/wind/nonlinear environment effects.
- MARS-style hinge discovery for thresholds such as high EV plus launch-angle windows, wind-out conditions, or extreme temperature.
- Park random effects in GLMM, with year/season fixed effects for league environment.

### Market Odds

The market is one of the strongest public predictors in sports. Baseball betting-market literature, including Woodland and Woodland on favorite-longshot bias and Brown and Abraham on MLB over/under efficiency, supports treating odds as highly informative but not perfectly efficient.

Repo mapping:

- Betting overlay: store raw odds, sportsbook, timestamp, market type, vig-free implied probability, and closing line.
- Direct theory implementation: market-offset GLM and complement model.
- Governance: report performance both with and without market offset so the structural model's contribution is visible.
- Avoid: training labels or features that smuggle in same-game closing information when the prediction is supposed to represent an earlier market snapshot.

## Model Patterns That Have Stood The Test

### 1. Component Expected-Runs Model

Build expected runs for each team from offense, starting pitching, bullpen, defense, park, weather, and lineup.

Recommended actuarial form:

- Negative binomial or overdispersed Poisson for team runs.
- Separate home and away expected-runs models, with shared context terms.
- GLMM random effects for teams, parks, pitchers, catchers, and season.
- GAM/MARS nonlinear terms for weather and Statcast contact-shape surfaces.

Why it survives:

- Baseball scoring is count data with overdispersion.
- Totals and runline need score distribution, not only win probability.
- Component models are easier to audit for leakage and baseball sanity.

### 2. Paired-Comparison Moneyline Model

Moneyline can be modeled as a logistic paired-comparison problem once each team's latent strength or expected run advantage is known.

Recommended actuarial form:

- Logistic GLM/GLMM with home advantage, expected run differential, market offset, and team/pitcher random effects.
- Bradley-Terry/log5-style matchup layer for team strength.
- DGLM or Elo-like state variable for evolving team quality.

Why it survives:

- It is interpretable on the log-odds scale.
- It naturally supports market offsets.
- It separates team strength estimation from binary outcome calibration.

### 3. Joint Score Distribution For Totals And Runline

Runline and totals should be derived from a joint run distribution when possible.

Recommended actuarial form:

- Independent NB as a baseline.
- Bivariate Poisson/NB or shared random-effect model when residual home/away scoring correlation is material.
- Skellam-style margin model as a compact challenger for run differential.
- Simulation layer from expected run distributions for alternate lines.

Why it survives:

- Totals are about `home_runs + away_runs`.
- Runline is about `home_runs - away_runs`.
- A moneyline-only model discards too much information for these markets.

### 4. Credibility-Weighted Ensemble

The final production system should be an ensemble only after the base learners have evidence.

Recommended actuarial form:

- Structural run model.
- Moneyline paired-comparison model.
- Market-offset residual model.
- Dynamic team/pitcher rating model.
- Optional Markov/simulation challenger.
- Penalized stacking or credibility-weighted blending, with written champion/challenger evidence.

Why it survives:

- Baseball has multiple weak-to-medium signals rather than one magic signal.
- Penalized ensembles are compatible with the repo's lasso-credibility lane.
- The market can be included without hiding whether the baseball model adds value.

## Leakage And Timestamp Rules

Pregame-only prediction means every feature must be available at or before the prediction timestamp.

Required timestamps:

- `game_id`
- `scheduled_start_time`
- `prediction_timestamp`
- `feature_snapshot_timestamp`
- `lineup_snapshot_timestamp`
- `weather_snapshot_timestamp`
- `market_snapshot_timestamp`
- `closing_line_timestamp`
- `label_timestamp`

Rules:

- Use announced lineups only after they are announced; otherwise use projected lineups and mark them as projected.
- Use forecast weather and announced roof state for pregame predictions, not observed in-game weather.
- Use market open/current/close according to the prediction contract; never allow a close price into an earlier prediction snapshot.
- Use rolling lagged player and team features. No same-game Statcast, injury, lineup, or bullpen usage can enter that game's pregame row.
- Keep raw market odds and vig-free transformations separate from model outputs.

## Validation And Promotion Evidence

Model promotion should require written evidence, not just a better-looking dashboard.

Minimum validation:

- Rolling-origin backtests split by game date.
- Separate train/validation/test windows, with a final untouched holdout period.
- Brier score, log loss, calibration plots, reliability tables, and sharpness.
- Moneyline ROI only after vig-free probability and price availability are audited.
- CLV/closing-line comparison as an external benchmark, not the sole target.
- Segment tables: home/away, favorite/underdog, park/weather buckets, starter quality buckets, totals bands, runline bands, and market-implied probability bands.
- Ablations: no-market model, market-only model, structural-plus-market model, and simple Marcel/Pythagorean baseline.

Promotion memo should answer:

- What is the champion?
- What baseline did it beat?
- Did it beat market-only or only baseball-only baselines?
- Is it calibrated?
- Does it hold across multiple seasons and segments?
- Which features are direct theory implementation, theory-compatible engineering support, betting overlay, and dashboard/reporting?
- What are the known failure modes?

## Concrete Repo Recommendations

### Feature Blocks

1. Lineup offense: projected PA-weighted wOBA/xwOBA, K%, BB%, ISO, barrel rate, platoon split, handedness mix, speed/baserunning.
2. Starter: K%, BB%, HR%, GB%, xwOBA allowed, barrel allowed, pitch velocity/movement/spin, expected innings, rest, injury state.
3. Bullpen: talent index, workload availability, handedness availability, leverage availability.
4. Defense/catcher: OAA/fielding run value, catcher framing/blocking/throwing, pitcher-catcher pair when known.
5. Park/weather: park run/HR factors, temperature, wind, humidity/pressure, altitude, roof, year run environment.
6. Schedule: home field, travel/rest, doubleheader, getaway day, time-zone changes, consecutive games.
7. Market: raw odds, de-vig probability, open/current/close snapshot, spread/runline/totals prices, market movement.

### Model Factory Priorities

1. NB expected-runs GLM for home/away runs.
2. Logistic moneyline GLM with expected run differential.
3. Market-offset logistic GLM.
4. GLMM random effects for team, park, pitcher, catcher, and season.
5. GAM weather/contact-quality nonlinear layer.
6. Elastic-net residual model with strict timestamped feature registry.
7. Joint score distribution challenger for runline/totals.
8. Champion ensemble only after written evidence.

### Baselines To Keep Forever

- Market-only vig-free probability.
- Marcel-style regressed team/player projection.
- Pythagorean expected win model from projected runs.
- Simple home-field plus starting-pitcher component model.
- No-market structural model.

## Source Notes

Key sources read or checked:

- Tom Tango, "The Marcel the Monkey Forecasting System" and "The 2004 Marcels": https://www.tangotiger.net/marcel/ and https://tangotiger.net/archives/stud0346.shtml
- MLB Glossary, wOBA: https://www.mlb.com/glossary/advanced-stats/weighted-on-base-average
- MLB Glossary, FIP: https://www.mlb.com/glossary/advanced-stats/fielding-independent-pitching
- MLB Glossary, BABIP: https://www.mlb.com/glossary/advanced-stats/babip
- MLB Glossary, park factor: https://www.mlb.com/glossary/advanced-stats/park-factor
- MLB Statcast Glossary, xwOBA: https://www.mlb.com/glossary/statcast/expected-woba
- MLB Statcast Glossary, xERA: https://www.mlb.com/glossary/statcast/expected-era
- MLB Statcast Glossary, barrel: https://www.mlb.com/glossary/statcast/barrel
- MLB Statcast Glossary, OAA: https://www.mlb.com/glossary/statcast/outs-above-average
- MLB Statcast Glossary, catcher framing: https://www.mlb.com/glossary/statcast/catcher-framing
- Baseball Savant, Statcast Metrics Context: https://baseballsavant.mlb.com/statcast-metrics-context
- Alan M. Nathan, "The Effect of Temperature on Home Run Production (Revisited)": https://baseball.physics.illinois.edu/HRProbTemp.pdf
- Eiji Konaka, "Park factor estimation improvement using pairwise comparison method": https://arxiv.org/abs/2109.09287
- Steven J. Miller, "A Derivation of the Pythagorean Won-Loss Formula in Baseball": https://arxiv.org/abs/math/0509698
- Victor Luo and Steven J. Miller, "Relieving and Readjusting Pythagoras": https://arxiv.org/abs/1406.3402
- Christopher Boudreaux, Justin Ehrlich, Shankar Ghimire, Shane Sanders, "Application of the Pythagorean Expected Wins Percentage and Cross-Validation Methods in Estimating Team Quality": https://arxiv.org/abs/2201.01168
- Clay Davenport and Keith Woolner, "Revisiting the Pythagorean Theorem": https://www.baseballprospectus.com/news/article/342/revisiting-the-pythagorean-theorem-putting-bill-james-pythagorean-theorem-to-the-test/
- Bruce Bukiet, Elliotte Rusty Harold, Jose Luis Palacios, "A Markov Chain Approach to Baseball": https://doi.org/10.1287/opre.45.1.14
- Jim Albert, "Beyond runs expectancy": https://doi.org/10.3233/jsa-140001
- Benjamin S. Baumer, Shane T. Jensen, Gregory J. Matthews, "openWAR": https://arxiv.org/abs/1312.7158
- Woodland and Woodland, "Market Efficiency and the Favorite-Longshot Bias: The Baseball Betting Market": https://doi.org/10.2307/2329144
- Brown and Abraham, "Testing Market Efficiency in the Major League Baseball Over-Under Betting Market": https://doi.org/10.1177/152700250200300401
- Erik Strumbelj, "On determining probability forecasts from betting odds": https://doi.org/10.1016/j.ijforecast.2014.02.008
- Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation": https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf

Some useful FanGraphs/Baseball Prospectus projection-system pages are Cloudflare-blocked from this environment, but the public technical pattern is consistent with the directly accessible Marcel, MLB, Statcast, arXiv, Crossref, and Baseball Prospectus sources above: weighted history, regression, age, component skills, park/context adjustment, and rigorous validation.
