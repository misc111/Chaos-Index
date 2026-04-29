"""Public facade for nested MLB tournament orchestration.

The implementation is split by responsibility across sibling modules while this
file preserves historical imports used by scripts, commands, and tests.
"""

from __future__ import annotations

import argparse

from src.common.config import load_config
from src.research.model_comparison import FEATURE_POOL_FULL_SCREENED
from src.research.nested_tournament_candidates import _candidate_specs
from src.research.nested_tournament_contracts import (
    CORE_LINEAR_FAMILIES,
    DEFAULT_NESTED_MODELS,
    DEFAULT_STRUCTURED_SPEC_PATH,
    EXTENSION_FAMILIES,
    MARKET_BASELINE_FAMILIES,
    FeatureVariant,
    NestedCandidateSpec,
    NestedTournamentResult,
    _safe_json,
    _utc_stamp,
)
from src.research.nested_tournament_governance import _gate_reasons
from src.research.nested_tournament_parallel import run_mlb_parallel_nested_tournament
from src.research.nested_tournament_sequential import run_mlb_nested_tournament

__all__ = [
    "CORE_LINEAR_FAMILIES",
    "DEFAULT_NESTED_MODELS",
    "DEFAULT_STRUCTURED_SPEC_PATH",
    "EXTENSION_FAMILIES",
    "MARKET_BASELINE_FAMILIES",
    "FeatureVariant",
    "NestedCandidateSpec",
    "NestedTournamentResult",
    "_candidate_specs",
    "_gate_reasons",
    "_safe_json",
    "_utc_stamp",
    "build_parser",
    "main",
    "run_mlb_nested_tournament",
    "run_mlb_parallel_nested_tournament",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the sequential nested tournament."""

    parser = argparse.ArgumentParser(description="Run the nested all-target MLB model tournament.")
    parser.add_argument("--config", default="configs/mlb.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--candidate-models", default=None)
    parser.add_argument("--feature-pool", default=FEATURE_POOL_FULL_SCREENED)
    parser.add_argument("--feature-map-model", default="glm_ridge")
    parser.add_argument("--structured-glm-spec", default=DEFAULT_STRUCTURED_SPEC_PATH)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sequential nested tournament from CLI arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = run_mlb_nested_tournament(
        cfg,
        run_id=args.run_id,
        targets=args.targets,
        candidate_models=args.candidate_models,
        feature_pool=args.feature_pool,
        feature_map_model=args.feature_map_model,
        structured_glm_spec_path=args.structured_glm_spec,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    print(f"MLB_NESTED_TOURNAMENT::{result.run_id}::{result.artifact_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
