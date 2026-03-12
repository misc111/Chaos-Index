"""Canonical subsystem metadata for generated docs and package READMEs."""

from __future__ import annotations

from src.registry.types import SubsystemDocEntry


SUBSYSTEM_DOCS: tuple[SubsystemDocEntry, ...] = (
    SubsystemDocEntry(
        path="src/bayes",
        title="Bayes",
        summary="Bayesian state-space fitting and daily update flows.",
        public_entrypoints=("src/bayes/fit_offline.py", "src/bayes/update_daily.py"),
        readme_path="src/bayes/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/commands",
        title="Commands",
        summary="Thin CLI wrappers that translate parsed arguments into service calls.",
        public_entrypoints=("src/cli.py", "src/commands/__init__.py"),
        readme_path="src/commands/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/common",
        title="Common",
        summary="Cross-cutting config, manifest, logging, time, and utility helpers.",
        public_entrypoints=("src/common/config.py", "src/common/manifests.py"),
        readme_path="src/common/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/data_sources",
        title="Data Sources",
        summary="League-specific ingest adapters and shared HTTP source contracts.",
        public_entrypoints=("src/data_sources/base.py", "src/league_registry.py"),
        readme_path="src/data_sources/README.md",
    ),
    SubsystemDocEntry(
        path="src/evaluation",
        title="Evaluation",
        summary="Scoring, diagnostics, drift checks, and validation artifact generation.",
        public_entrypoints=("src/evaluation/validation_pipeline.py", "src/services/validate.py"),
        readme_path="src/evaluation/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/features",
        title="Features",
        summary="League-aware feature engineering over shared pipeline stages.",
        public_entrypoints=("src/features/build_features.py", "src/features/pipeline.py"),
        readme_path="src/features/README.md",
    ),
    SubsystemDocEntry(
        path="src/models",
        title="Models",
        summary="Model implementations for rating, linear, tree, simulation, and Bayesian families.",
        public_entrypoints=("src/models/base.py", "src/training/model_catalog.py"),
        readme_path="src/models/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/orchestration",
        title="Orchestration",
        summary="Deterministic multi-league refresh and publish pipelines.",
        public_entrypoints=("src/orchestration/data_refresh.py", "src/orchestration/hard_refresh.py"),
        readme_path="src/orchestration/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/query",
        title="Query",
        summary="Deterministic natural-language query handling for local model/product questions.",
        public_entrypoints=("src/query/answer.py", "src/query/intent_parser.py"),
        readme_path="src/query/README.md",
    ),
    SubsystemDocEntry(
        path="src/research",
        title="Research",
        summary="Research-only comparison and experimentation flows over candidate model sets.",
        public_entrypoints=("src/research/model_comparison.py", "src/research/candidate_models.py"),
        readme_path="src/research/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/services",
        title="Services",
        summary="Application-layer orchestration for ingest, training, validation, and backtests.",
        public_entrypoints=("src/services/ingest.py", "src/services/train.py"),
        readme_path="src/services/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/simulation",
        title="Simulation",
        summary="Simulation-specific helpers used by forecast and betting workflows.",
        public_entrypoints=("src/simulation/game_simulator.py",),
        readme_path="src/simulation/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/storage",
        title="Storage",
        summary="SQLite schema, query helpers, and persistence contracts.",
        public_entrypoints=("src/storage/db.py", "src/storage/schema.py"),
        readme_path="src/storage/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="src/training",
        title="Training",
        summary="Training orchestration, feature policy, ensembles, and prediction runners.",
        public_entrypoints=("src/training/train.py", "src/training/model_catalog.py"),
        readme_path="src/training/README.md",
    ),
    SubsystemDocEntry(
        path="web/app/api",
        title="Web API",
        summary="Thin Next.js route handlers over server-side repositories and services.",
        public_entrypoints=("web/app/api/predictions/route.ts", "web/lib/server/services"),
        readme_path="web/app/api/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="web/lib/hooks",
        title="Web Hooks",
        summary="Client-side React hooks for league, strategy, and dashboard state wiring.",
        public_entrypoints=("web/lib/hooks/useLeague.ts", "web/lib/hooks/useDashboardData.ts"),
        readme_path="web/lib/hooks/README.md",
        generate_readme=True,
    ),
    SubsystemDocEntry(
        path="web/lib/server",
        title="Web Server Lib",
        summary="Server-only repositories and route services behind the dashboard API.",
        public_entrypoints=("web/lib/server/manifests.ts", "web/lib/server/services"),
        readme_path="web/lib/server/README.md",
    ),
    SubsystemDocEntry(
        path="configs/generated",
        title="Generated Config",
        summary="Generated manifests consumed by Python, the dashboard, and docs verification.",
        public_entrypoints=("configs/generated/league_manifest.json", "configs/generated/model_manifest.json"),
        readme_path="configs/generated/README.md",
        generate_readme=True,
    ),
)


def subsystem_docs() -> tuple[SubsystemDocEntry, ...]:
    """Return the canonical subsystem documentation registry."""

    return SUBSYSTEM_DOCS
