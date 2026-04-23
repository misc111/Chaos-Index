#!/usr/bin/env zsh
set -euo pipefail

python3 -m src.cli init-db --config configs/mlb.yaml
python3 -m src.cli fetch --config configs/mlb.yaml
python3 -m src.cli features --config configs/mlb.yaml
python3 -m src.cli train --config configs/mlb.yaml
python3 -m src.cli backtest --config configs/mlb.yaml
python3 -m src.query.answer --config configs/mlb.yaml --question "What's the chance the Dodgers win their next game?"
python3 -m src.query.answer --config configs/mlb.yaml --question "Which model has performed best the last 60 days?"
