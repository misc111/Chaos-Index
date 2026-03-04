SHELL := /bin/zsh
PYTHON ?= python3
PIP ?= pip3
NPM ?= npm
MODELS ?=
MODEL_ARGS := $(if $(MODELS),--models "$(MODELS)",)

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  install-python      Install Python deps"
	@echo "  install-node        Install Node deps"
	@echo "  init-db             Initialize SQLite schema"
	@echo "  fetch               Fetch NHL data"
	@echo "  features            Build feature tables"
	@echo "  train               Train models + predict upcoming games"
	@echo "  backtest            Walk-forward backtest + artifacts"
	@echo "                      Optional: MODELS=glm_logit,rf (default: all)"
	@echo "  run_daily           Daily pipeline end-to-end"
	@echo "  dashboard           Launch Next.js dashboard"
	@echo "  query Q=...         Query local forecast/performance DB"
	@echo "  smoke               End-to-end smoke run"
	@echo "  test                Run tests"

install-python:
	$(PIP) install -e '.[dev]'

install-node:
	cd web && $(NPM) install

init-db:
	$(PYTHON) -m src.cli init-db --config configs/nhl.yaml

fetch:
	$(PYTHON) -m src.cli fetch --config configs/nhl.yaml

features:
	$(PYTHON) -m src.cli features --config configs/nhl.yaml

train:
	$(PYTHON) -m src.cli train --config configs/nhl.yaml $(MODEL_ARGS)

backtest:
	$(PYTHON) -m src.cli backtest --config configs/nhl.yaml $(MODEL_ARGS)

run_daily:
	$(PYTHON) -m src.cli run-daily --config configs/nhl.yaml $(MODEL_ARGS)

dashboard:
	cd web && $(NPM) run dev

query:
	$(PYTHON) -m src.query.answer --config configs/nhl.yaml --question "$(Q)"

smoke:
	$(PYTHON) -m src.cli smoke --config configs/nhl.yaml

test:
	pytest
