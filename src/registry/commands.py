"""Canonical CLI command registry shared by argparse, docs, and help text."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from src.registry.leagues import default_config_path
from src.registry.types import CommandArgumentSpec, CommandRegistryEntry


def _option(*flags: str, doc_metavar: str | None = None, **argparse_kwargs: object) -> CommandArgumentSpec:
    """Create a declarative command argument spec."""

    return CommandArgumentSpec(flags=tuple(flags), argparse_kwargs=dict(argparse_kwargs), doc_metavar=doc_metavar)


_MODELS_ARG = _option(
    "--models",
    default="all",
    help="Comma-separated model list (for example: glm_ridge,rf) or 'all'.",
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
    help="Optional research-only YAML spec defining a structured NBA GLM feature slate.",
    doc_metavar="PATH",
)
_STRUCTURED_GLM_SLATE_ARG = _option(
    "--structured-glm-slate",
    default=None,
    help="Optional named slate to select from the structured NBA GLM spec.",
    doc_metavar="SLATE",
)
_STRUCTURED_GLM_WIDTH_VARIANT_ARG = _option(
    "--structured-glm-width-variant",
    default=None,
    help="Optional width variant to select from the structured NBA GLM spec.",
    doc_metavar="VARIANT",
)
_BOOTSTRAP_SAMPLES_ARG = _option(
    "--bootstrap-samples",
    type=int,
    default=1000,
    help="Number of paired bootstrap samples for the final holdout comparison.",
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
_MODEL_RUN_ID_ARG = _option(
    "--model-run-id",
    default=None,
    help="Optional saved base model run id to validate.",
    doc_metavar="RUN_ID",
)


COMMAND_REGISTRY: tuple[CommandRegistryEntry, ...] = (
    CommandRegistryEntry(
        name="init-db",
        summary="Initialize the SQLite schema for the selected league config.",
        handler_path="src.commands.data:init_db",
        examples=("make init-db", "make init-db CONFIG=configs/nhl.yaml"),
    ),
    CommandRegistryEntry(
        name="fetch",
        summary="Fetch league data, persist snapshots, and ingest results.",
        handler_path="src.commands.data:fetch",
        examples=("make fetch", "make fetch CONFIG=configs/ncaam.yaml"),
    ),
    CommandRegistryEntry(
        name="refresh-data",
        summary="Run the league-scoped data refresh flow including the final odds pull.",
        handler_path="src.commands.data:refresh_data",
        examples=("make refresh-data CONFIG=configs/nba.yaml",),
    ),
    CommandRegistryEntry(
        name="fetch-odds",
        summary="Fetch the latest standalone odds snapshot for the selected league.",
        handler_path="src.commands.data:fetch_odds",
        examples=("make fetch-odds", "make fetch-odds CONFIG=configs/nhl.yaml"),
    ),
    CommandRegistryEntry(
        name="import-history",
        summary="Import historical source snapshots into the local research dataset.",
        handler_path="src.commands.data:import_history",
        arguments=(_HISTORY_SEASONS_ARG, _SOURCE_MANIFEST_ARG),
        examples=("python3 -m src.cli import-history --config configs/nba.yaml --history-seasons 3",),
    ),
    CommandRegistryEntry(
        name="features",
        summary="Build processed feature tables from the current interim snapshot.",
        handler_path="src.commands.data:features",
        examples=("make features", "make features CONFIG=configs/ncaam.yaml"),
    ),
    CommandRegistryEntry(
        name="research-features",
        summary="Score and optionally promote per-model feature maps.",
        handler_path="src.commands.data:research_features",
        arguments=(_MODELS_ARG, _APPROVE_FEATURE_CHANGES_ARG),
        examples=("make research-features CONFIG=configs/nba.yaml APPROVE_FEATURE_CHANGES=1",),
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
        examples=("make train", "make train CONFIG=configs/nba.yaml MODELS=glm_ridge,rf"),
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
        examples=("make validate", "make validate CONFIG=configs/nba.yaml MODEL_RUN_ID=run_abc123"),
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
        examples=("make compare-candidates CONFIG=configs/nba.yaml",),
    ),
    CommandRegistryEntry(
        name="backtest",
        summary="Run the walk-forward backtest and scoring pipeline.",
        handler_path="src.commands.modeling:backtest",
        arguments=(_MODELS_ARG, _APPROVE_FEATURE_CHANGES_ARG),
        examples=("make backtest", "make backtest CONFIG=configs/ncaam.yaml MODELS=glm_ridge"),
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
        examples=("python3 -m src.cli research-backtest --config configs/nba.yaml --history-seasons 2",),
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
        examples=("make run_daily", "make run_daily CONFIG=configs/nba.yaml MODELS=glm_ridge"),
    ),
    CommandRegistryEntry(
        name="smoke",
        summary="Exercise the local end-to-end smoke pipeline with reduced data windows.",
        handler_path="src.commands.smoke:run",
        examples=("make smoke",),
    ),
)

_COMMAND_BY_NAME = {entry.name: entry for entry in COMMAND_REGISTRY}


def command_registry() -> tuple[CommandRegistryEntry, ...]:
    """Return all canonical CLI command definitions."""

    return COMMAND_REGISTRY


def command_names() -> tuple[str, ...]:
    """Return the canonical CLI command name tuple."""

    return tuple(entry.name for entry in COMMAND_REGISTRY)


def get_command_spec(name: str) -> CommandRegistryEntry:
    """Resolve a command name into registry metadata."""

    return _COMMAND_BY_NAME[name]


def get_command_handler(name: str) -> Callable[..., object]:
    """Resolve the callable handler for a registered command."""

    spec = get_command_spec(name)
    module_name, attribute_name = spec.handler_path.split(":", 1)
    module = importlib.import_module(module_name)
    return cast(Callable[..., object], getattr(module, attribute_name))


def command_manifest_payload() -> dict[str, object]:
    """Render the deterministic command manifest payload."""

    return {
        "version": 1,
        "source": "code_registry",
        "default_config_path": default_config_path("NBA"),
        "commands": [
            {
                "name": entry.name,
                "summary": entry.summary,
                "handler_path": entry.handler_path,
                "config_required": entry.config_required,
                "arguments": [
                    {
                        "flags": list(argument.flags),
                        "doc_metavar": argument.doc_metavar,
                        "help": argument.argparse_kwargs.get("help"),
                        "choices": list(argument.argparse_kwargs["choices"])
                        if "choices" in argument.argparse_kwargs
                        else None,
                        "default": argument.argparse_kwargs.get("default"),
                    }
                    for argument in entry.arguments
                ],
                "examples": list(entry.examples),
            }
            for entry in COMMAND_REGISTRY
        ],
    }


def render_make_help() -> str:
    """Render the dynamic CLI section for `make help`."""

    lines = [
        "CLI-backed targets:",
    ]
    for entry in COMMAND_REGISTRY:
        lines.append(f"  {entry.name:<18} {entry.summary}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_make_help())
