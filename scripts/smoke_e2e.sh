#!/usr/bin/env zsh
set -euo pipefail

python3 -m src.cli init-db --config configs/nhl.yaml
python3 -m src.cli fetch --config configs/nhl.yaml
python3 -m src.cli features --config configs/nhl.yaml
python3 -m src.cli train --config configs/nhl.yaml
python3 -m src.cli backtest --config configs/nhl.yaml
python3 -m src.query.answer --config configs/nhl.yaml --question "What's the chance the Leafs win their next game?"
python3 -m src.query.answer --config configs/nhl.yaml --question "Which model has performed best the last 60 days?"
