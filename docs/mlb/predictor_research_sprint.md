# MLB Predictor Research Sprint

This note turns outside baseball modeling references into testable MLB tournament variant ideas. It is not a promotion memo. Every idea below must still survive the nested tournament gates before it can influence a champion decision.

## Source-Backed Predictor Families

### Team, Starter, Travel, Rest

FiveThirtyEight's MLB methodology describes a baseball rating model that adjusts pregame team strength for home field, travel, rest, park/era context, and especially the scheduled starting pitcher. This supports keeping team-strength, rest/travel, and starter-quality terms in the theory-core GLM lane, with starter-specific variants tested inside each family.

Implementation stance:

- Keep `team_strength_edge`, `starter_quality_edge`, `rest_edge`, `travel_miles_edge`, and `timezone_edge` in core/stability variants.
- Add starter-first variants that can absorb richer projected starter quality when upstream features exist.
- Treat opener/bulk-pitcher uncertainty as an explicit uncertainty term, not silent noise.

Source:

- FiveThirtyEight, "How Our MLB Predictions Work": https://fivethirtyeight.com/methodology/how-our-mlb-predictions-work/

### Fielding-Independent Pitching And Pitcher Shape

FanGraphs' pitcher library emphasizes FIP/FIP- for defense-independent pitcher performance, and xFIP as a more expectation-oriented pitcher metric that normalizes home-run-per-fly-ball volatility. MLB's SIERA glossary adds strikeout, walk, and batted-ball shape through grounders, fly balls, and pop-ups.

Implementation stance:

- Add a `starter_run_prevention` slate in the structured GLM config.
- Prefer projected or rolling pregame versions of `starter_xfip_edge`, `starter_fip_minus_edge`, `starter_siera_edge`, `starter_k_rate_edge`, `starter_bb_rate_edge`, and batted-ball/contact suppression terms.
- Do not mix raw ERA-only terms into champion slates without a guardrail note, because fielding/park/luck contamination is exactly what these metrics try to reduce.

Sources:

- FanGraphs, "FIP": https://library.fangraphs.com/pitching/fip/
- FanGraphs, "xFIP": https://library.fangraphs.com/pitching/xfip/
- MLB Glossary, "Skill-interactive Earned Run Average": https://www.mlb.com/glossary/advanced-stats/skill-interactive-earned-run-average

### Offense, wRC+, And Contact Quality

FanGraphs describes wRC+ as a park- and league-adjusted offensive rate metric based on wOBA. Baseball Savant exposes Statcast contact and expected-outcome concepts such as exit velocity, launch angle, barrels, hard-hit balls, xBA, xwOBA, pitch velocity, movement, spin, extension, xERA, fielding run value, catcher blocking, and sprint speed.

Implementation stance:

- Add a `lineup_platoon_contact` slate for confirmed-lineup quality, platoon-adjusted offense, xwOBA/contact quality, strikeout/chase profile, and pitch-mix matchup terms.
- Keep lineup uncertainty separate from lineup talent so the model can learn lower confidence when lineups are not confirmed.
- Treat Statcast shape variables as theory-compatible engineering support unless they are transformed into GLM-stable rate/edge terms with leakage checks.

Sources:

- FanGraphs, "wRC and wRC+": https://library.fangraphs.com/offense/wrc/
- Baseball Savant / Statcast glossary surface: https://baseballsavant.mlb.com/

### Park, Weather, And Run Environment

FanGraphs' park-factor explanation stresses that dimensions alone are insufficient; weather, air density, altitude, local topology, and handedness-specific effects matter. MLB's wind analysis likewise highlights that wind can materially change high-air-ball outcomes, especially now that in-park weather measurement is available.

Implementation stance:

- Keep `weather_context` as a broad moneyline/runline environment test.
- Add a `totals_environment` slate centered on park, weather, roof, altitude, umpire, offense, starter suppression, bullpen fatigue, and market-total context.
- For totals, prefer run-environment features over generic home-win features; do not imply moneyline validation proves totals skill.

Sources:

- FanGraphs, "Park Factors": https://library.fangraphs.com/principles/park-factors/
- MLB.com, "The big impact of wind on baseball outcomes": https://www.mlb.com/news/the-big-impact-of-wind-on-baseball-outcomes

### Bullpen, Catcher, Umpire, And Market Context

Public projection systems such as THE BAT X advertise game-level inputs that include opposing hitter/pitcher context, ballpark, weather, umpire, catcher framing/throwing, bullpen, pitch counts, home field, platoon splits, defense, lineup position, and surrounding lineup quality. This is not a transparent academic source, but it is a useful checklist for feature backlog completeness.

Implementation stance:

- Add a `bullpen_workload` slate for bullpen quality, recent pitch count, back-to-back usage, high-leverage usage, rest days, and starter expected innings.
- Keep catcher/umpire features as extension variants until source coverage and pregame availability are proven.
- Keep vig-free market probabilities and line movement in explicitly market-aware variants or offset/complement lanes, not hidden inside theory-core predictors.

Source:

- THE BAT X public feature list: https://rotogrinders.com/the-bat

## Engineering Backlog

1. Build upstream pregame feature columns for the new config names, or map current repository feature names into these aliases.
2. Add feature-source metadata: projection, rolling historical, Statcast shape, market, weather, lineup, or fixture.
3. Extend leakage checks so starter, bullpen, lineup, weather, and market features must be known before first pitch.
4. Add missing-line/low-coverage gates for runline and totals market-dependent variants.
5. Run the nested tournament by target, keeping these slates as variants rather than merging them into one large specification.

## Current Config Changes

The structured GLM tournament config now includes these research-backed slates:

- `starter_run_prevention`
- `bullpen_workload`
- `lineup_platoon_contact`
- `totals_environment`

The tournament runner already filters absent features out of a slate, so these variants become active only as upstream pregame feature coverage matures.
