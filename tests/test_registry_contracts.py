import argparse
import json

from src.cli import build_parser
from src.registry.commands import command_manifest_payload, command_names
from src.registry.dashboard_routes import dashboard_route_manifest_payload
from src.registry.generate import ROOT_DIR, generate_all
from src.registry.leagues import league_manifest_payload
from src.registry.models import model_manifest_payload
from src.registry.subsystems import subsystem_docs
from src.training.model_catalog import (
    ALL_MODEL_NAMES,
    EXPERIMENTAL_MODEL_NAMES,
    GOVERNANCE_COMPARISON_GROUPS,
    MODEL_ALIASES,
    MODEL_REPORT_ORDER,
    THEORY_EXTENSION_MODEL_NAMES,
)


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


def test_mlb_train_cli_defaults_to_the_core_lane() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "configs/mlb.yaml"])

    assert args.models is None


def test_structured_glm_cli_flags_are_exposed_on_research_commands() -> None:
    parser = build_parser()

    compare_args = parser.parse_args(
        [
            "compare-candidates",
            "--structured-glm-spec",
            "configs/research/nba_glm_rewrite_v1.yaml",
            "--structured-glm-slate",
            "core_market_form",
            "--structured-glm-width-variant",
            "wide",
        ]
    )
    assert compare_args.structured_glm_spec == "configs/research/nba_glm_rewrite_v1.yaml"
    assert compare_args.structured_glm_slate == "core_market_form"
    assert compare_args.structured_glm_width_variant == "wide"

    backtest_args = parser.parse_args(
        [
            "research-backtest",
            "--structured-glm-spec",
            "configs/research/nba_glm_rewrite_v1.yaml",
            "--structured-glm-slate",
            "pace_and_pressure",
            "--structured-glm-width-variant",
            "narrow",
        ]
    )
    assert backtest_args.structured_glm_spec == "configs/research/nba_glm_rewrite_v1.yaml"
    assert backtest_args.structured_glm_slate == "pace_and_pressure"
    assert backtest_args.structured_glm_width_variant == "narrow"


def test_generated_league_manifest_matches_code_registry() -> None:
    manifest = json.loads((ROOT_DIR / "configs" / "generated" / "league_manifest.json").read_text())

    assert manifest == league_manifest_payload()
    assert manifest["primary_league"] == "MLB"
    assert manifest["primary_rebuild_leagues"] == ["MLB"]
    assert manifest["leagues"]["MLB"]["primary_rebuild_lane"] is True
    assert manifest["leagues"]["MLB"]["lifecycle"] == "primary"
    assert sorted(manifest["leagues"]) == ["MLB"]


def test_generated_model_manifest_matches_code_registry_and_training_catalog() -> None:
    manifest = json.loads((ROOT_DIR / "configs" / "generated" / "model_manifest.json").read_text())

    assert manifest == model_manifest_payload()
    assert manifest["primary_lane"] == "core"
    assert manifest["trainable_models"] == ALL_MODEL_NAMES
    assert manifest["default_training_models"] == [
        "glm_ridge",
        "glm_elastic_net",
        "glm_lasso",
        "glm_vanilla",
    ]
    assert manifest["core_models"] == [
        "glm_ridge",
        "glm_elastic_net",
        "glm_lasso",
        "glm_lasso_market_credibility",
        "glm_lasso_prior_credibility",
        "glm_vanilla",
    ]
    assert manifest["theory_extension_models"] == [
        "gam_spline",
        "glmm_logit",
        "dglm_margin",
        "goals_poisson",
    ]
    assert manifest["baseline_models"] == []
    assert manifest["experimental_models"] == EXPERIMENTAL_MODEL_NAMES
    assert manifest["aliases"] == MODEL_ALIASES
    assert manifest["prediction_report_order"] == MODEL_REPORT_ORDER
    assert manifest["theory_extension_models"] == THEORY_EXTENSION_MODEL_NAMES
    assert manifest["lane_labels"]["experimental"] == "Experimental challenger lane"
    assert manifest["models"]["gam_spline"]["lane"] == "extension"
    assert manifest["models"]["glm_lasso_market_credibility"]["default_enabled"] is False
    assert manifest["models"]["glm_lasso_prior_credibility"]["default_enabled"] is False
    assert manifest["models"]["gam_spline"]["default_enabled"] is False
    assert manifest["models"]["glmm_logit"]["default_enabled"] is False
    assert manifest["models"]["mars_hinge"]["lane"] == "experimental"
    assert manifest["models"]["bayes_bt_state_space"]["implementation_namespace"] == "src.models.experimental"


def test_prediction_report_order_keeps_core_rows_ahead_of_baseline_and_experimental() -> None:
    manifest = model_manifest_payload()
    ordered = [name for name in manifest["prediction_report_order"] if name != "ensemble"]
    lane_priority = {"core": 0, "extension": 1, "baseline": 2, "experimental": 3}
    lane_sequence = [lane_priority[manifest["models"][name]["lane"]] for name in ordered]

    assert lane_sequence == sorted(lane_sequence)
    assert GOVERNANCE_COMPARISON_GROUPS["theory_core_default"]["model_keys"] == manifest["default_training_models"]
    assert GOVERNANCE_COMPARISON_GROUPS["experimental_challengers"]["champion_eligible"] is False


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
