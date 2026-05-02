# Training

This subsystem turns processed features into model artifacts, forecasts, scores, and validation outputs.

MLB product lane:
- `src/training/train.py` is the orchestration entry point.
- `src/training/feature_contract.py` owns deterministic run-wide and model-specific feature-column resolution for training, backtests, and width evaluation.
- Model metadata lives in `src/training/model_catalog.py`, with fit/predict/ensemble/uncertainty/artifact responsibilities split into dedicated modules.
- The canonical model manifest is loaded through `src/common/manifests.py` so Python and the web app read the same catalog.
- CAS-governed core models import from `src.models.core.*`, while non-CAS challengers import from `src.models.experimental.*`.

Legacy/comparison boundary:
- NBA/NHL training lanes are not runtime defaults. Recover them from git history only for explicit comparison work.
- MLB-specific behavior should stay in named policies or payload builders, not inline in orchestration.

Input/output contract:
- Input is a processed feature frame plus config-selected training options.
- Output is persisted model artifacts, OOF/prediction tables, backtest-ready scores, and web-facing payload inputs with stable model names from the shared manifest.
