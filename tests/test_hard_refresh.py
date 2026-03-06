import sys

import pytest

from src.orchestration.hard_refresh import ROOT_DIR, build_hard_refresh_steps


def test_build_hard_refresh_steps_default_sequence():
    steps = build_hard_refresh_steps()

    assert [step.name for step in steps] == [
        "nhl:init-db",
        "nhl:fetch",
        "nhl:fetch-odds",
        "nhl:features",
        "nhl:train",
        "nba:init-db",
        "nba:fetch",
        "nba:fetch-odds",
        "nba:features",
        "nba:train",
        "staging:generate-data",
        "staging:build-pages",
    ]
    assert steps[0].command == (sys.executable, "-m", "src.cli", "init-db", "--config", "configs/nhl.yaml")
    assert steps[4].command == (sys.executable, "-m", "src.cli", "train", "--config", "configs/nhl.yaml")
    assert steps[10].cwd == ROOT_DIR / "web"


def test_build_hard_refresh_steps_models_and_approve_flag():
    steps = build_hard_refresh_steps(models_arg="glm,rf,glm", approve_feature_changes=True, include_pages_build=False)

    train_steps = [step for step in steps if step.name.endswith(":train")]
    assert len(train_steps) == 2
    for step in train_steps:
        assert step.command[-3:] == ("--models", "glm_logit,rf", "--approve-feature-changes")

    assert [step.name for step in steps][-1] == "staging:generate-data"


def test_build_hard_refresh_steps_rejects_unknown_models():
    with pytest.raises(ValueError):
        build_hard_refresh_steps(models_arg="not_a_model")
