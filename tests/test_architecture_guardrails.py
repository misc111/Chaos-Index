from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.registry.generate import ROOT_DIR
from src.registry.leagues import league_manifest_payload
from src.registry.models import (
    default_training_model_names,
    ordered_model_entries,
    theory_extension_model_names,
    validate_model_registry_contract,
)
from src.registry.verify import _check_primary_staging_contract
from src.research.artifact_guardrails import require_mlb_report_path, require_mlb_tournament_path
from src.research.mlb_model_tournament import THEORY_CLASSIFICATION_BY_MODEL


ALLOWED_THEORY_LABELS = {"core-supported", "theory-compatible extension", "experimental"}
TOP_LEVEL_MODEL_IMPLEMENTATIONS = {
    "__init__.py",
    "base.py",
}
CORE_MODEL_IMPLEMENTATIONS = {
    "__init__.py",
    "glm_penalized.py",
    "glm_vanilla.py",
    "lasso_credibility.py",
}
CORE_COMPATIBILITY_SHIM_MODULES = {
    "src.models.glm_elastic_net",
    "src.models.glm_lasso",
    "src.models.glm_penalized",
    "src.models.glm_ridge",
    "src.models.lasso_credibility",
}


def _has_class_or_function(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    return any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_model_registry_enforces_core_extension_experimental_boundaries() -> None:
    entries = {entry.key: entry for entry in ordered_model_entries()}
    default_models = set(default_training_model_names())
    extension_models = set(theory_extension_model_names())

    assert default_models == {"glm_ridge", "glm_elastic_net", "glm_lasso", "glm_vanilla"}
    assert not default_models.intersection(extension_models)

    for entry in entries.values():
        assert entry.theory_classification in ALLOWED_THEORY_LABELS, entry.key
        if entry.lane == "core":
            assert entry.theory_classification == "core-supported", entry.key
            assert entry.implementation_namespace == "src.models.core", entry.key
        if entry.lane == "extension":
            assert entry.theory_classification == "theory-compatible extension", entry.key
            assert entry.implementation_namespace == "src.models.extensions", entry.key
            assert entry.default_enabled is False, entry.key
        if entry.lane == "experimental":
            assert entry.theory_classification == "experimental", entry.key
            assert entry.implementation_namespace == "src.models.experimental", entry.key
            assert entry.default_enabled is False, entry.key
        if entry.lane == "baseline":
            assert entry.theory_classification == "experimental", entry.key
            assert entry.default_enabled is False, entry.key

    assert entries["mars_hinge"].lane == "experimental"
    assert entries["two_stage"].lane == "experimental"
    assert entries["gam_spline"].lane == "extension"


def test_model_registry_contract_blocks_default_non_core_drift() -> None:
    entries = {entry.key: entry for entry in ordered_model_entries()}

    assert validate_model_registry_contract() == []
    for model_name in default_training_model_names():
        entry = entries[model_name]
        assert entry.lane == "core"
        assert entry.theory_classification == "core-supported"
        assert entry.default_enabled is True


def test_top_level_models_are_core_implementations_or_shims_only() -> None:
    model_root = ROOT_DIR / "src" / "models"
    for path in sorted(model_root.glob("*.py")):
        if path.name in TOP_LEVEL_MODEL_IMPLEMENTATIONS:
            continue
        assert not _has_class_or_function(path), f"{path.relative_to(ROOT_DIR)} must be a compatibility shim"
        text = path.read_text()
        assert "Compatibility shim" in text, f"{path.relative_to(ROOT_DIR)} must declare shim status"


def test_core_models_live_in_the_explicit_core_namespace() -> None:
    core_root = ROOT_DIR / "src" / "models" / "core"
    implementation_files = sorted(core_root.glob("*.py"))

    assert {path.name for path in implementation_files} == CORE_MODEL_IMPLEMENTATIONS
    for path in implementation_files:
        if path.name == "__init__.py":
            continue
        assert _has_class_or_function(path), f"{path.relative_to(ROOT_DIR)} must define a core implementation"


def test_runtime_code_imports_core_models_from_the_core_namespace() -> None:
    src_root = ROOT_DIR / "src"
    model_root = src_root / "models"
    shim_paths = {model_root / f"{module.rsplit('.', 1)[-1]}.py" for module in CORE_COMPATIBILITY_SHIM_MODULES}

    for path in sorted(src_root.rglob("*.py")):
        if path in shim_paths:
            continue
        stale_imports = _imported_modules(path).intersection(CORE_COMPATIBILITY_SHIM_MODULES)
        assert not stale_imports, f"{path.relative_to(ROOT_DIR)} imports core models through top-level shims"


def test_primary_staging_contract_exposes_only_mlb_at_the_root() -> None:
    staging_root = ROOT_DIR / "web" / "public" / "staging-data"
    manifest = json.loads((staging_root / "manifest.json").read_text())
    root_dirs = sorted(path.name for path in staging_root.iterdir() if path.is_dir())

    assert manifest["primary_league"] == "MLB"
    assert manifest["shipped_leagues"] == ["MLB"]
    assert manifest["leagues"] == ["MLB"]
    assert sorted(manifest["required_files_by_league"]) == ["MLB"]
    assert root_dirs == ["mlb"]
    assert not (staging_root / "legacy").exists()
    assert _check_primary_staging_contract() == []


def test_mlb_artifact_write_guards_reject_history_and_non_mlb_lanes(tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"

    good_report = artifacts_dir / "reports" / "mlb" / "scorecard.json"
    good_tournament = artifacts_dir / "reports" / "mlb" / "tournament" / "run_1"
    assert require_mlb_report_path(good_report, artifacts_dir, purpose="unit test") == good_report
    assert require_mlb_tournament_path(good_tournament, artifacts_dir, purpose="unit test") == good_tournament

    with pytest.raises(RuntimeError, match="artifacts/reports/mlb"):
        require_mlb_report_path(artifacts_dir / "reports" / "history" / "scorecard.json", artifacts_dir, purpose="unit test")

    with pytest.raises(RuntimeError, match="artifacts/reports/mlb/tournament"):
        require_mlb_tournament_path(artifacts_dir / "reports" / "mlb" / "scorecard.json", artifacts_dir, purpose="unit test")


def test_only_mlb_is_registered_as_a_supported_league() -> None:
    manifest = league_manifest_payload()

    assert manifest["primary_league"] == "MLB"
    assert manifest["primary_rebuild_leagues"] == ["MLB"]
    assert sorted(manifest["leagues"]) == ["MLB"]
    assert manifest["leagues"]["MLB"]["lifecycle"] == "primary"
    assert sorted(manifest["legacy_comparison_leagues"]) == ["NBA", "NHL"]
    assert "NBA" not in manifest["leagues"]
    assert "NHL" not in manifest["leagues"]


def test_leak_repaired_tournament_outputs_retain_theory_labels_when_present() -> None:
    artifact_root = (
        ROOT_DIR
        / "artifacts"
        / "reports"
        / "mlb"
        / "tournament"
        / "research_director_20260422_second_round_leakage_repaired"
    )

    assert set(THEORY_CLASSIFICATION_BY_MODEL) >= {"glm_elastic_net", "glm_ridge", "glm_lasso", "mars_hinge"}
    if not artifact_root.exists():
        return

    leaderboard = json.loads((artifact_root / "leaderboard.json").read_text())
    recommendation = json.loads((artifact_root / "recommendation.json").read_text())

    assert leaderboard, "leak-repaired leaderboard exists but is empty"
    assert {row.get("theory_classification") for row in leaderboard} <= ALLOWED_THEORY_LABELS
    assert all(row.get("theory_classification") for row in leaderboard)
    assert {row.get("theory_classification") for row in recommendation["recommendations"]} <= ALLOWED_THEORY_LABELS
    assert all(row.get("theory_classification") for row in recommendation["recommendations"])
    assert (artifact_root / "theory_compliance_summary.md").exists()
