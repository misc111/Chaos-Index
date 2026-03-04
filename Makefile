SHELL := /bin/zsh
PYTHON ?= python3
PIP ?= pip3
NPM ?= npm
MODELS ?=
MODEL_ARGS := $(if $(MODELS),--models "$(MODELS)",)
CONFIG ?= configs/nhl.yaml

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  install-python      Install Python deps"
	@echo "  install-node        Install Node deps"
	@echo "  init-db             Initialize SQLite schema"
	@echo "  fetch               Fetch league data from CONFIG"
	@echo "  features            Build feature tables"
	@echo "  train               Train models + predict upcoming games"
	@echo "  backtest            Walk-forward backtest + artifacts"
	@echo "                      Optional: MODELS=glm_logit,rf (default: all)"
	@echo "  run_daily           Daily pipeline end-to-end"
	@echo "  dashboard           Launch Next.js dashboard"
	@echo "  query Q=...         Query local forecast/performance DB"
	@echo "  smoke               End-to-end smoke run"
	@echo "  test                Run tests"
	@echo "  "
	@echo "Usage:"
	@echo "  make fetch CONFIG=configs/nba.yaml"
	@echo "  make query CONFIG=configs/nba.yaml Q=\"What's the chance the Raptors win the next game?\""

install-python:
	$(PIP) install -e '.[dev]'

install-node:
	cd web && $(NPM) install

init-db:
	$(PYTHON) -m src.cli init-db --config $(CONFIG)

fetch:
	$(PYTHON) -m src.cli fetch --config $(CONFIG)

features:
	$(PYTHON) -m src.cli features --config $(CONFIG)

train:
	$(PYTHON) -m src.cli train --config $(CONFIG) $(MODEL_ARGS)

backtest:
	$(PYTHON) -m src.cli backtest --config $(CONFIG) $(MODEL_ARGS)

run_daily:
	$(PYTHON) -m src.cli run-daily --config $(CONFIG) $(MODEL_ARGS)

dashboard:
	cd web && $(NPM) run dev

query:
	$(PYTHON) -m src.query.answer --config $(CONFIG) --question "$(Q)"

smoke:
	$(PYTHON) -m src.cli smoke --config $(CONFIG)

test:
	pytest
