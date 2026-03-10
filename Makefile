SHELL := /bin/zsh
PYTHON ?= python3
PIP ?= pip3
NPM ?= npm
MODELS ?=
MODEL_ARGS := $(if $(MODELS),--models "$(MODELS)",)
MODEL_RUN_ID ?=
MODEL_RUN_ARGS := $(if $(MODEL_RUN_ID),--model-run-id "$(MODEL_RUN_ID)",)
APPROVE_FEATURE_CHANGES ?= 0
APPROVE_FEATURE_ARGS := $(if $(filter 1 true TRUE yes YES,$(APPROVE_FEATURE_CHANGES)),--approve-feature-changes,)
PAGES_BUILD ?= 1
PAGES_BUILD_ARGS := $(if $(filter 0 false FALSE no NO,$(PAGES_BUILD)),--skip-pages-build,)
DRY_RUN ?= 0
DRY_RUN_ARGS := $(if $(filter 1 true TRUE yes YES,$(DRY_RUN)),--dry-run,)
CONFIG ?= configs/nba.yaml

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  install-python      Install Python deps"
	@echo "  install-node        Install Node deps"
	@echo "  init-db             Initialize SQLite schema"
	@echo "  fetch               Fetch league data from CONFIG"
	@echo "  refresh-data        Fetch league data + final odds pull from CONFIG"
	@echo "  fetch-odds          Fetch latest odds snapshot from CONFIG"
	@echo "  features            Build feature tables"
	@echo "  research-features   Score and promote per-model feature maps"
	@echo "  train               Train models + predict upcoming games"
	@echo "                      Optional: MODELS=glm_ridge,rf (default: all)"
	@echo "                      Optional: APPROVE_FEATURE_CHANGES=1"
	@echo "  validate            Regenerate validation artifacts from the latest saved trained run"
	@echo "                      Optional: MODELS=glm_ridge,rf (default: saved run selection)"
	@echo "                      Optional: MODEL_RUN_ID=run_abc123"
	@echo "  compare-candidates  Run the research-only candidate model comparison suite"
	@echo "                      Optional: CONFIG=configs/nba.yaml"
	@echo "                      Optional: CANDIDATE_MODELS=glm_ridge,glm_lasso,glm_elastic_net,glm_vanilla FEATURE_POOL=production_model_map FEATURE_MAP_MODEL=glm_ridge"
	@echo "  backtest            Walk-forward backtest + artifacts"
	@echo "                      Optional: MODELS=glm_ridge,rf (default: all)"
	@echo "                      Optional: APPROVE_FEATURE_CHANGES=1"
	@echo "  run_daily           Daily pipeline end-to-end"
	@echo "  data_refresh        Deterministic multi-league data-only refresh"
	@echo "                      Optional: DRY_RUN=1"
	@echo "  hard_refresh        Deterministic multi-league refresh/train + staging snapshot"
	@echo "                      Uses existing processed features; does not rebuild features"
	@echo "                      Optional: MODELS=glm_ridge,rf APPROVE_FEATURE_CHANGES=1 PAGES_BUILD=0 DRY_RUN=1"
	@echo "  dashboard           Launch Next.js dashboard"
	@echo "  smoke-dashboard     Playwright smoke test for the Next.js dashboard"
	@echo "  query Q=...         Query local forecast/performance DB"
	@echo "  smoke               End-to-end smoke run"
	@echo "  test                Run tests"
	@echo "  "
	@echo "Usage:"
	@echo "  make fetch CONFIG=configs/nba.yaml"
	@echo "  make refresh-data CONFIG=configs/nba.yaml"
	@echo "  make data_refresh DRY_RUN=1"
	@echo "  make query CONFIG=configs/nba.yaml Q=\"What's the chance the Raptors win the next game?\""

install-python:
	$(PIP) install -e '.[dev]'

install-node:
	cd web && $(NPM) install

init-db:
	$(PYTHON) -m src.cli init-db --config $(CONFIG)

fetch:
	$(PYTHON) -m src.cli fetch --config $(CONFIG)

refresh-data:
	$(PYTHON) -m src.cli refresh-data --config $(CONFIG)

fetch-odds:
	$(PYTHON) -m src.cli fetch-odds --config $(CONFIG)

features:
	$(PYTHON) -m src.cli features --config $(CONFIG)

research-features:
	$(PYTHON) -m src.cli research-features --config $(CONFIG) $(MODEL_ARGS) $(APPROVE_FEATURE_ARGS)

train:
	$(PYTHON) -m src.cli train --config $(CONFIG) $(MODEL_ARGS) $(APPROVE_FEATURE_ARGS)

validate:
	$(PYTHON) -m src.cli validate --config $(CONFIG) $(MODEL_ARGS) $(MODEL_RUN_ARGS)

compare-candidates:
	$(PYTHON) -m src.cli compare-candidates --config $(CONFIG) $(if $(CANDIDATE_MODELS),--candidate-models "$(CANDIDATE_MODELS)",) $(if $(FEATURE_POOL),--feature-pool "$(FEATURE_POOL)",) $(if $(FEATURE_MAP_MODEL),--feature-map-model "$(FEATURE_MAP_MODEL)",)

backtest:
	$(PYTHON) -m src.cli backtest --config $(CONFIG) $(MODEL_ARGS) $(APPROVE_FEATURE_ARGS)

run_daily:
	$(PYTHON) -m src.cli run-daily --config $(CONFIG) $(MODEL_ARGS) $(APPROVE_FEATURE_ARGS)

data_refresh:
	$(PYTHON) -m src.orchestration.data_refresh $(DRY_RUN_ARGS)

hard_refresh:
	$(PYTHON) -m src.orchestration.hard_refresh $(MODEL_ARGS) $(APPROVE_FEATURE_ARGS) $(PAGES_BUILD_ARGS) $(DRY_RUN_ARGS)

dashboard:
	cd web && $(NPM) run dev

smoke-dashboard:
	cd web && $(NPM) run playwright:install && $(NPM) run test:smoke

query:
	$(PYTHON) -m src.query.answer --config $(CONFIG) --question "$(Q)"

smoke:
	$(PYTHON) -m src.cli smoke --config $(CONFIG)

test:
	pytest
