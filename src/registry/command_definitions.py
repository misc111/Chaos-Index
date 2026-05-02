"""Declarative CLI command definitions for the MLB-first command registry.

This module intentionally owns the verbose command table and argument specs so
``src.registry.commands`` can stay focused on resolution, manifests, and help
rendering.  The definitions remain data-only: no handler imports happen here, so
registry generation can inspect commands without importing operational flows.
"""

from __future__ import annotations

from src.registry.types import CommandArgumentSpec, CommandRegistryEntry

def _option(*flags: str, doc_metavar: str | None = None, **argparse_kwargs: object) -> CommandArgumentSpec:
    """Create a declarative command argument spec."""

    return CommandArgumentSpec(flags=tuple(flags), argparse_kwargs=dict(argparse_kwargs), doc_metavar=doc_metavar)


_MODELS_ARG = _option(
    "--models",
    default=None,
    help="Comma-separated model list (for example: glm_ridge,rf) or 'all'. Omit to use the MLB core-supported GLM lane.",
    doc_metavar="MODELS",
)
_VALIDATE_MODELS_ARG = _option(
    "--models",
    default=None,
    help="Comma-separated model list (for example: glm_ridge,rf) or 'all'.",
    doc_metavar="MODELS",
)
_APPROVE_FEATURE_CHANGES_ARG = _option(
    "--approve-feature-changes",
    action="store_true",
    help="Explicitly accept and persist model feature-contract changes.",
)
_VALIDATION_MODE_ARG = _option(
    "--validation-split-mode",
    choices=("train_test", "train_validation_test"),
    default=None,
    help="Validation split layout: 70/30 train-test or 40/30/30 train-validation-test.",
    doc_metavar="MODE",
)
_VALIDATION_METHOD_ARG = _option(
    "--validation-split-method",
    choices=("time", "random"),
    default=None,
    help="Validation split method: out-of-time or random-by-record.",
    doc_metavar="METHOD",
)
_VALIDATION_SEED_ARG = _option(
    "--validation-split-seed",
    type=int,
    default=None,
    help="Optional random seed for random-by-record validation splits.",
    doc_metavar="SEED",
)
_REPORT_SLUG_ARG = _option(
    "--report-slug",
    default=None,
    help="Optional report slug prefix for artifact outputs.",
    doc_metavar="SLUG",
)
_CANDIDATE_MODELS_ARG = _option(
    "--candidate-models",
    default="all",
    help="Comma-separated candidate model list or 'all'.",
    doc_metavar="MODELS",
)
_FEATURE_POOL_ARG = _option(
    "--feature-pool",
    choices=("full_screened", "production_model_map", "research_broad"),
    default="full_screened",
    help="Feature pool for the comparison flow.",
    doc_metavar="POOL",
)
_FEATURE_MAP_MODEL_ARG = _option(
    "--feature-map-model",
    default="glm_ridge",
    help="Model key to read from the production feature map when needed.",
    doc_metavar="MODEL",
)
_STRUCTURED_GLM_SPEC_ARG = _option(
    "--structured-glm-spec",
    default=None,
    help="Optional research-only YAML spec defining a structured MLB GLM feature slate.",
    doc_metavar="PATH",
)
_STRUCTURED_GLM_SLATE_ARG = _option(
    "--structured-glm-slate",
    default=None,
    help="Optional named slate to select from the structured MLB GLM spec.",
    doc_metavar="SLATE",
)
_STRUCTURED_GLM_WIDTH_VARIANT_ARG = _option(
    "--structured-glm-width-variant",
    default=None,
    help="Optional width variant to select from the structured MLB GLM spec.",
    doc_metavar="VARIANT",
)
_BOOTSTRAP_SAMPLES_ARG = _option(
    "--bootstrap-samples",
    type=int,
    default=1000,
    help="Number of paired bootstrap samples for the final holdout comparison.",
    doc_metavar="N",
)
_TARGETS_ARG = _option(
    "--targets",
    default="all",
    help="Comma-separated nested tournament targets: moneyline_home_win,runline_home_cover,totals_over, or 'all'.",
    doc_metavar="TARGETS",
)
_RUN_ID_ARG = _option(
    "--run-id",
    default=None,
    help="Optional run identifier override.",
    doc_metavar="RUN_ID",
)
_PARALLEL_ARG = _option(
    "--parallel",
    action="store_true",
    help="Evaluate intra-family target/model lanes in parallel before central final-holdout ranking.",
)
_MAX_WORKERS_ARG = _option(
    "--max-workers",
    type=int,
    default=4,
    help="Maximum parallel intra-family lane workers.",
    doc_metavar="N",
)
_HISTORY_SEASONS_ARG = _option(
    "--history-seasons",
    type=int,
    default=None,
    help="Override the configured number of historical seasons.",
    doc_metavar="N",
)
_SOURCE_MANIFEST_ARG = _option(
    "--source-manifest",
    default=None,
    help="Optional absolute or config-relative path to the historical import manifest.",
    doc_metavar="PATH",
)
_START_DATE_ARG = _option(
    "--start-date",
    default=None,
    help="Optional inclusive start date for historical odds backfill (YYYY-MM-DD).",
    doc_metavar="DATE",
)
_END_DATE_ARG = _option(
    "--end-date",
    default=None,
    help="Optional inclusive end date for historical odds backfill (YYYY-MM-DD).",
    doc_metavar="DATE",
)
_CHUNK_DAYS_ARG = _option(
    "--chunk-days",
    type=int,
    default=30,
    help="Number of dates per historical odds backfill chunk.",
    doc_metavar="N",
)
_BRIEF_ARG = _option(
    "--brief",
    default=None,
    help="Optional brief key or path for the research desk flow.",
    doc_metavar="BRIEF",
)
_BRIEF_DIR_ARG = _option(
    "--brief-dir",
    default=None,
    help="Optional directory containing structured research briefs.",
    doc_metavar="PATH",
)
_MODEL_RUN_ID_ARG = _option(
    "--model-run-id",
    default=None,
    help="Optional saved base model run id to validate.",
    doc_metavar="RUN_ID",
)
_SEASON_ARG = _option(
    "--season",
    type=int,
    default=None,
    help="Optional MLB season override for current-season evidence reports.",
    doc_metavar="SEASON",
)


COMMAND_REGISTRY: tuple[CommandRegistryEntry, ...] = (
    CommandRegistryEntry(
        name="init-db",
        summary="Initialize the SQLite schema for the MLB config.",
        handler_path="src.commands.data:init_db",
        examples=("make init-db", "make init-db CONFIG=configs/mlb.yaml"),
    ),
    CommandRegistryEntry(
        name="fetch",
        summary="Fetch MLB data, persist snapshots, and ingest results.",
        handler_path="src.commands.data:fetch",
        examples=("make fetch", "make fetch CONFIG=configs/mlb.yaml"),
    ),
    CommandRegistryEntry(
        name="refresh-data",
        summary="Run the MLB data refresh flow including the final odds pull.",
        handler_path="src.commands.data:refresh_data",
        examples=("make refresh-data CONFIG=configs/mlb.yaml",),
    ),
    CommandRegistryEntry(
        name="fetch-odds",
        summary="Fetch the latest standalone MLB odds snapshot.",
        handler_path="src.commands.data:fetch_odds",
        examples=("make fetch-odds", "make fetch-odds CONFIG=configs/mlb.yaml"),
    ),
    CommandRegistryEntry(
        name="import-history",
        summary="Import historical source snapshots into the local research dataset.",
        handler_path="src.commands.data:import_history",
        arguments=(_HISTORY_SEASONS_ARG, _SOURCE_MANIFEST_ARG),
        examples=("python3 -m src.cli import-history --config configs/mlb.yaml --history-seasons 3",),
    ),
    CommandRegistryEntry(
        name="backfill-historical-odds",
        summary="Download historical odds bundles into a regenerable manifest-backed cache.",
        handler_path="src.commands.data:backfill_historical_odds",
        arguments=(_HISTORY_SEASONS_ARG, _START_DATE_ARG, _END_DATE_ARG, _CHUNK_DAYS_ARG),
        examples=(
            "python3 -m src.cli backfill-historical-odds --config configs/mlb.yaml --start-date 2025-03-20 --end-date 2025-09-30",
        ),
    ),
    CommandRegistryEntry(
        name="features",
        summary="Build processed feature tables from the current interim snapshot.",
        handler_path="src.commands.data:features",
        examples=("make features", "make features CONFIG=configs/mlb.yaml"),
    ),
    CommandRegistryEntry(
        name="research-features",
        summary="Score and optionally promote per-model feature maps.",
        handler_path="src.commands.data:research_features",
        arguments=(_MODELS_ARG, _APPROVE_FEATURE_CHANGES_ARG),
        examples=("make research-features CONFIG=configs/mlb.yaml APPROVE_FEATURE_CHANGES=1",),
    ),
    CommandRegistryEntry(
        name="train",
        summary="Train models, produce forecasts, and persist validation-ready outputs.",
        handler_path="src.commands.modeling:train",
        arguments=(
            _MODELS_ARG,
            _VALIDATION_MODE_ARG,
            _VALIDATION_METHOD_ARG,
            _VALIDATION_SEED_ARG,
            _APPROVE_FEATURE_CHANGES_ARG,
        ),
        examples=("make train", "make train CONFIG=configs/mlb.yaml MODELS=glm_ridge,glm_lasso"),
    ),
    CommandRegistryEntry(
        name="validate",
        summary="Regenerate validation artifacts from the latest saved trained run.",
        handler_path="src.commands.modeling:validate",
        arguments=(
            _VALIDATE_MODELS_ARG,
            _VALIDATION_MODE_ARG,
            _VALIDATION_METHOD_ARG,
            _VALIDATION_SEED_ARG,
            _MODEL_RUN_ID_ARG,
        ),
        examples=("make validate", "make validate CONFIG=configs/mlb.yaml MODEL_RUN_ID=run_abc123"),
    ),
    CommandRegistryEntry(
        name="compare-candidates",
        summary="Run the research-only candidate model comparison suite.",
        handler_path="src.commands.modeling:compare_candidates",
        arguments=(
            _REPORT_SLUG_ARG,
            _BOOTSTRAP_SAMPLES_ARG,
            _CANDIDATE_MODELS_ARG,
            _FEATURE_POOL_ARG,
            _FEATURE_MAP_MODEL_ARG,
            _STRUCTURED_GLM_SPEC_ARG,
            _STRUCTURED_GLM_SLATE_ARG,
            _STRUCTURED_GLM_WIDTH_VARIANT_ARG,
        ),
        examples=("make compare-candidates CONFIG=configs/mlb.yaml",),
    ),
    CommandRegistryEntry(
        name="best-models",
        summary="Print the canonical current-best-models summary for the MLB research lane.",
        handler_path="src.commands.modeling:best_models",
        examples=("python3 -m src.cli best-models --config configs/mlb.yaml",),
    ),
    CommandRegistryEntry(
        name="current-season-predictiveness",
        summary="Score frozen pregame MLB predictions against current-season settled outcomes.",
        handler_path="src.commands.modeling:current_season_predictiveness",
        arguments=(
            _REPORT_SLUG_ARG,
            _SEASON_ARG,
        ),
        examples=("python3 -m src.cli current-season-predictiveness --config configs/mlb.yaml",),
    ),
    CommandRegistryEntry(
        name="nested-tournament",
        summary="Run the nested all-target MLB tournament with intra-family selection before inter-family final holdout.",
        handler_path="src.commands.modeling:nested_tournament",
        arguments=(
            _RUN_ID_ARG,
            _TARGETS_ARG,
            _BOOTSTRAP_SAMPLES_ARG,
            _CANDIDATE_MODELS_ARG,
            _FEATURE_POOL_ARG,
            _FEATURE_MAP_MODEL_ARG,
            _STRUCTURED_GLM_SPEC_ARG,
            _PARALLEL_ARG,
            _MAX_WORKERS_ARG,
        ),
        examples=("python3 -m src.cli nested-tournament --config configs/mlb.yaml --targets all",),
    ),
    CommandRegistryEntry(
        name="feature-availability",
        summary="Write structured MLB feature availability coverage artifacts for tournament slates.",
        handler_path="src.commands.modeling:feature_availability",
        arguments=(
            _RUN_ID_ARG,
            _STRUCTURED_GLM_SPEC_ARG,
        ),
        examples=("python3 -m src.cli feature-availability --config configs/mlb.yaml",),
    ),
    CommandRegistryEntry(
        name="backtest",
        summary="Run the walk-forward backtest and scoring pipeline.",
        handler_path="src.commands.modeling:backtest",
        arguments=(_MODELS_ARG, _APPROVE_FEATURE_CHANGES_ARG),
        examples=("make backtest", "make backtest CONFIG=configs/mlb.yaml MODELS=glm_ridge"),
    ),
    CommandRegistryEntry(
        name="research-backtest",
        summary="Run the research backtest over a historical candidate-model dataset.",
        handler_path="src.commands.modeling:research_backtest",
        arguments=(
            _REPORT_SLUG_ARG,
            _CANDIDATE_MODELS_ARG,
            _FEATURE_POOL_ARG,
            _FEATURE_MAP_MODEL_ARG,
            _HISTORY_SEASONS_ARG,
            _STRUCTURED_GLM_SPEC_ARG,
            _STRUCTURED_GLM_SLATE_ARG,
            _STRUCTURED_GLM_WIDTH_VARIANT_ARG,
        ),
        examples=("python3 -m src.cli research-backtest --config configs/mlb.yaml --history-seasons 2",),
    ),
    CommandRegistryEntry(
        name="research-desk",
        summary="Run the MLB-first research desk orchestration flow with structured brief intake and champion auto-promotion.",
        handler_path="src.commands.modeling:research_desk",
        arguments=(
            _REPORT_SLUG_ARG,
            _BRIEF_ARG,
            _BRIEF_DIR_ARG,
            _CANDIDATE_MODELS_ARG,
            _FEATURE_POOL_ARG,
            _FEATURE_MAP_MODEL_ARG,
            _HISTORY_SEASONS_ARG,
            _STRUCTURED_GLM_SPEC_ARG,
            _STRUCTURED_GLM_SLATE_ARG,
            _STRUCTURED_GLM_WIDTH_VARIANT_ARG,
        ),
        examples=("make research_desk CONFIG=configs/mlb.yaml",),
    ),
    CommandRegistryEntry(
        name="run-daily",
        summary="Execute the daily fetch, feature, train, and scoring flow.",
        handler_path="src.commands.modeling:run_daily",
        arguments=(
            _MODELS_ARG,
            _VALIDATION_MODE_ARG,
            _VALIDATION_METHOD_ARG,
            _VALIDATION_SEED_ARG,
            _APPROVE_FEATURE_CHANGES_ARG,
        ),
        examples=("make run_daily", "make run_daily CONFIG=configs/mlb.yaml MODELS=glm_ridge"),
    ),
    CommandRegistryEntry(
        name="smoke",
        summary="Exercise the local end-to-end smoke pipeline with reduced data windows.",
        handler_path="src.commands.smoke:run",
        examples=("make smoke",),
    ),
)
