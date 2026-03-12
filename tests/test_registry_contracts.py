import argparse
import json

from src.cli import build_parser
from src.registry.commands import command_manifest_payload, command_names
from src.registry.dashboard_routes import dashboard_route_manifest_payload
from src.registry.generate import ROOT_DIR, generate_all
from src.registry.leagues import league_manifest_payload
from src.registry.models import model_manifest_payload
from src.registry.subsystems import subsystem_docs
from src.training.model_catalog import ALL_MODEL_NAMES, MODEL_ALIASES, MODEL_REPORT_ORDER


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("CLI parser did not register subcommands.")


def test_generated_registry_artifacts_are_current() -> None:
    assert generate_all(check=True) == []


def test_cli_parser_stays_in_sync_with_command_registry() -> None:
    parser = build_parser()
    subparser_action = _subparser_action(parser)

    assert tuple(subparser_action.choices.keys()) == command_names()


def test_generated_league_manifest_matches_code_registry() -> None:
    manifest = json.loads((ROOT_DIR / "configs" / "generated" / "league_manifest.json").read_text())

    assert manifest == league_manifest_payload()


def test_generated_model_manifest_matches_code_registry_and_training_catalog() -> None:
    manifest = json.loads((ROOT_DIR / "configs" / "generated" / "model_manifest.json").read_text())

    assert manifest == model_manifest_payload()
    assert manifest["trainable_models"] == ALL_MODEL_NAMES
    assert manifest["aliases"] == MODEL_ALIASES
    assert manifest["prediction_report_order"] == MODEL_REPORT_ORDER


def test_generated_command_manifest_matches_code_registry() -> None:
    manifest = json.loads((ROOT_DIR / "configs" / "generated" / "command_manifest.json").read_text())

    assert manifest == command_manifest_payload()


def test_generated_dashboard_route_manifest_matches_code_registry_and_real_files() -> None:
    manifest = json.loads((ROOT_DIR / "configs" / "generated" / "dashboard_route_manifest.json").read_text())
    route_files = set()
    staging_files = set()

    assert manifest == dashboard_route_manifest_payload()

    for route in manifest["routes"]:
        route_path = ROOT_DIR / "web" / route["module_path"]
        assert route_path.exists()
        assert route["key"] == route["payload_contract"]
        assert route["module_path"] not in route_files
        assert route["staging_file_name"] not in staging_files
        route_files.add(route["module_path"])
        staging_files.add(route["staging_file_name"])


def test_generated_subsystem_readmes_exist_for_all_registered_subsystems() -> None:
    for entry in subsystem_docs():
        if entry.readme_path is None:
            continue
        assert (ROOT_DIR / entry.readme_path).exists(), entry.readme_path
