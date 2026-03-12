"""Repo-level contract verification for generated surfaces and architecture rules."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

from src.registry.generate import ROOT_DIR, generate_all
from src.registry.subsystems import subsystem_docs


REQUIRED_READMES = tuple(
    ROOT_DIR / entry.readme_path
    for entry in subsystem_docs()
    if entry.readme_path is not None
)
PUBLIC_DOCSTRING_TARGETS = (
    ROOT_DIR / "src/cli.py",
    ROOT_DIR / "src/league_registry.py",
    ROOT_DIR / "src/common/manifests.py",
    ROOT_DIR / "src/commands/__init__.py",
    ROOT_DIR / "src/commands/data.py",
    ROOT_DIR / "src/commands/modeling.py",
    ROOT_DIR / "src/commands/smoke.py",
    ROOT_DIR / "src/orchestration/data_refresh.py",
    ROOT_DIR / "src/orchestration/hard_refresh.py",
    ROOT_DIR / "src/orchestration/refresh_pipeline.py",
    ROOT_DIR / "src/training/model_catalog.py",
    ROOT_DIR / "src/training/model_feature_guardrails.py",
    ROOT_DIR / "src/query/answer.py",
    ROOT_DIR / "src/registry/types.py",
    ROOT_DIR / "src/registry/leagues.py",
    ROOT_DIR / "src/registry/models.py",
    ROOT_DIR / "src/registry/commands.py",
    ROOT_DIR / "src/registry/dashboard_routes.py",
    ROOT_DIR / "src/registry/subsystems.py",
    ROOT_DIR / "src/registry/generate.py",
)
OVERSIZED_FILE_ALLOWLIST = {
    "src/research/model_comparison.py",
    "src/evaluation/validation_pipeline.py",
    "src/evaluation/validation_stability.py",
    "src/evaluation/validation_classification.py",
    "src/research/candidate_models.py",
    "src/training/model_feature_research.py",
    "src/evaluation/diagnostics_glm.py",
    "src/services/ingest.py",
    "src/services/history_import.py",
    "src/services/research_backtest.py",
    "src/features/strategies/nba.py",
    "src/data_sources/odds_api.py",
    "src/evaluation/validation_nonlinearity.py",
    "src/query/bet_history_handlers.py",
    "src/services/train.py",
    "web/lib/server/services/performance.ts",
    "web/lib/bet-history.ts",
    "web/lib/betting.ts",
    "web/components/bet-sizing/BetSizingExperience.tsx",
    "web/lib/replay-bets.ts",
    "web/components/EnsembleSnapshotBankrollChart.tsx",
    "web/lib/ensemble-snapshot-chart.test.ts",
    "web/lib/betting-optimizer.ts",
    "web/components/EnsembleSnapshotExplorer.tsx",
    "web/app/games-today/page.tsx",
    "web/components/DashboardHeader.tsx",
    "web/components/BetHistoryChart.tsx",
    "web/lib/ensemble-snapshot-replay.ts",
}
FORBIDDEN_LITERAL_PATTERNS = (
    re.compile(r"configs/(nhl|nba|ncaam)\.yaml"),
    re.compile(r"data/processed/(nhl|nba|ncaam)_forecast\.db"),
    re.compile(r"process\.env\.(NHL|NBA|NCAAM)_DB_PATH"),
    re.compile(r'process\.env\["(NHL|NBA|NCAAM)_DB_PATH"\]'),
)
FORBIDDEN_LITERAL_ALLOWLIST = {
    ROOT_DIR / "src/registry/leagues.py",
    ROOT_DIR / "src/registry/commands.py",
    ROOT_DIR / "src/registry/generate.py",
    ROOT_DIR / "web/lib/generated/league-registry.ts",
}
SOURCE_GLOBS = ("src/**/*.py", "web/**/*.ts", "web/**/*.tsx", "web/**/*.mjs")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SOURCE_GLOBS:
        files.extend(ROOT_DIR.glob(pattern))
    return sorted(
        path
        for path in files
        if "node_modules" not in path.parts and ".next" not in path.parts and "out" not in path.parts
    )


def _check_generated_artifacts() -> list[str]:
    return [str(path.relative_to(ROOT_DIR)) for path in generate_all(check=True)]


def _check_required_readmes() -> list[str]:
    return [str(path.relative_to(ROOT_DIR)) for path in REQUIRED_READMES if not path.exists()]


def _check_public_docstrings() -> list[str]:
    failures: list[str] = []
    for path in PUBLIC_DOCSTRING_TARGETS:
        source = path.read_text()
        tree = ast.parse(source)
        if ast.get_docstring(tree) is None:
            failures.append(f"{path.relative_to(ROOT_DIR)}: missing module docstring")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    failures.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno} missing public docstring for {node.name}")
    return failures


def _check_oversized_files(max_lines: int = 400) -> list[str]:
    failures: list[str] = []
    for path in _iter_source_files():
        relative = str(path.relative_to(ROOT_DIR))
        if relative.startswith("web/lib/generated/"):
            continue
        if path.suffix not in {".py", ".ts", ".tsx"}:
            continue
        line_count = sum(1 for _ in path.open())
        if line_count > max_lines and relative not in OVERSIZED_FILE_ALLOWLIST:
            failures.append(f"{relative}: {line_count} lines")
    return failures


def _check_forbidden_literals() -> list[str]:
    failures: list[str] = []
    for path in _iter_source_files():
        if path in FORBIDDEN_LITERAL_ALLOWLIST:
            continue
        relative = path.relative_to(ROOT_DIR)
        if str(relative).startswith("web/lib/generated/"):
            continue
        text = path.read_text()
        for pattern in FORBIDDEN_LITERAL_PATTERNS:
            match = pattern.search(text)
            if match:
                failures.append(f"{relative}:{match.start()} matched {pattern.pattern}")
    return failures


def main() -> None:
    """Run repo-level zero-drift verification checks."""

    parser = argparse.ArgumentParser(description="Verify generated artifacts and repo architecture contracts.")
    parser.parse_args()

    failures: list[str] = []

    stale_artifacts = _check_generated_artifacts()
    if stale_artifacts:
        failures.append("Generated artifacts are stale:")
        failures.extend(f"  - {item}" for item in stale_artifacts)

    missing_readmes = _check_required_readmes()
    if missing_readmes:
        failures.append("Missing required READMEs:")
        failures.extend(f"  - {item}" for item in missing_readmes)

    docstring_failures = _check_public_docstrings()
    if docstring_failures:
        failures.append("Missing public docstrings:")
        failures.extend(f"  - {item}" for item in docstring_failures)

    oversized_files = _check_oversized_files()
    if oversized_files:
        failures.append("Files above the line-count threshold without allowlisting:")
        failures.extend(f"  - {item}" for item in oversized_files)

    forbidden_literals = _check_forbidden_literals()
    if forbidden_literals:
        failures.append("Forbidden config/DB/env literals found outside resolver modules:")
        failures.extend(f"  - {item}" for item in forbidden_literals)

    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
