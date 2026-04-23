# MLB Feature Contract

## Governing Rule

All core features must be pregame-legal and explicitly timestamped or derivable as of the model's prediction timestamp.

## Availability Rule

Each feature family must declare or imply:

- raw source
- entity grain
- earliest reliable availability
- backfill behavior
- leakage risk notes

`available_as_of` must be enforced before the feature enters a model matrix.

Classification: `Theory-compatible engineering support`

## Current Sprint Status

Implemented in the current Sprint 2 foundation slice:

- `build_features_from_interim(..., league="MLB")` now resolves to an MLB strategy
- first-pass MLB feature families include:
  - run-form and run-differential history
  - starter ERA / WHIP proxies
  - lineup availability and confirmation
  - bullpen availability proxies
  - Elo and dynamic-rating priors
  - venue-based travel/rest context carried through the shared pipeline
- the processed feature frame now emits `available_as_of_utc` as an explicit contract column
- MLB travel features use MLB team-city mappings and compute travel from prior game venue to current game venue without cross-sport fallbacks

This is still an early MLB feature store, not the full target inventory. Weather, market movement, probable-starter depth, bullpen fatigue/usage detail, and richer lineup uncertainty still need to be layered in.

## Core MLB Pregame Feature Families

- team strength
- starting pitcher strength
- bullpen quality
- bullpen fatigue / recent usage
- lineup quality
- lineup availability / uncertainty
- recent offense
- recent defense / run prevention
- park effects
- home/away
- rest days
- travel
- timezone/circadian mismatch where supportable
- day/night
- weather
- season phase
- division / league effects
- market open / current / movement dispersion
- bookmaker dispersion
- time-to-first-pitch

Classification: `Direct theory implementation` once admitted into a model; `Theory-compatible engineering support` at raw-feature staging time

## Transform Policy

Allow:

- raw
- logged
- polynomial
- binned
- piecewise linear / hinge
- natural cubic spline
- grouped categories
- interactions

Only keep transforms that survive validation and stability review.

Classification: `Direct theory implementation`

## Lasso Credibility Feature Policy

Prefer:

- categorical variables
- ordinal variables
- explicit complements
- carefully chosen control variables

Use caution with:

- continuous-variable lasso credibility
- heavily correlated transform stacks
- opaque imported rating layers that weaken interpretability

Classification: `Direct theory implementation`
