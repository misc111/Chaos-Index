# Training

This subsystem turns processed features into model artifacts, forecasts, scores, and validation outputs.

Shared across leagues:
- `src/training/train.py` is the orchestration entry point.
- Model metadata lives in `src/training/model_catalog.py`, with fit/predict/ensemble/uncertainty/artifact responsibilities split into dedicated modules.
- The canonical model manifest is loaded through `src/common/manifests.py` so Python and the web app read the same catalog.

League-specific:
- League-aware behavior should stay in named policies or payload builders, not inline in orchestration.
- Examples include uncertainty flags, championship naming, and any future league-specific artifact shaping.

Input/output contract:
- Input is a processed feature frame plus config-selected training options.
- Output is persisted model artifacts, OOF/prediction tables, backtest-ready scores, and web-facing payload inputs with stable model names from the shared manifest.
