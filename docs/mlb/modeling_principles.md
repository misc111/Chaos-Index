# MLB Modeling Principles

## Governing Rule

The statistical core of the MLB rebuild is governed by the two CAS monographs. The default theory lane is:

1. vanilla GLM
2. penalized GLM
3. lasso credibility
4. GLM extensions
5. disciplined ensemble comparison

## Principles

### 1. Pregame only

All core features must be knowable before first pitch. Every feature must carry an explicit availability concept and be validated against `available_as_of`.

Classification: `Theory-compatible engineering support`

### 2. Targets before overlays

We model baseball outcomes first:

- home win
- runline cover
- totals over
- home runs
- away runs
- total runs
- run differential

Only after those theory-core targets are credible do we add betting overlays like edge, EV, and stake sizing.

Classification: `Direct theory implementation` for target modeling, `Betting overlay` for decisioning.

### 3. Right family, right link, right offset

Every model must record:

- target
- distribution
- link
- weights
- offsets
- feature set
- split metadata

Offsets must be transformed onto the same linear-predictor scale as the model.

Classification: `Direct theory implementation`

### 4. Market is a complement, not a substitute

The market-aware lane should use vig-free market probabilities and proper link-scale offsets to express complements of credibility. The product must preserve:

- market-free discovery lane
- market-offset credibility lane

Classification: `Direct theory implementation`

### 5. Ordinal and categorical clarity for lasso credibility

Where lasso credibility is used, prefer categorical and ordinal encodings that make shrinkage toward the complement visible and reviewable. Continuous-variable lasso credibility should be treated cautiously and justified explicitly.

Classification: `Direct theory implementation`

### 6. Validation is not optional

No model is real in this repo until it emits:

- fit statistics
- residual diagnostics
- calibration/lift evidence
- multicollinearity review
- stability evidence
- artifact manifest entries

Classification: `Direct theory implementation`

### 7. Champion promotion needs a written record

The final champion ensemble must have a promotion/rejection report that covers:

- serious candidates considered
- exact features used
- target/distribution/link
- complements and offsets
- validation results
- betting outcomes
- stability flags
- rejection reasons

Classification: `Theory-compatible engineering support`

### 8. Non-CAS models are challengers only

Tree models, neural nets, and Bayesian models may exist as experimental challengers, but they are not the default theory lane and must not silently become the main program.

Classification: `Betting overlay` or experimental support, not the core theory contract
