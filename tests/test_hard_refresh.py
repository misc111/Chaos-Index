import sys
import subprocess

import pytest

from src.orchestration import hard_refresh as hard_refresh_module
from src.orchestration.hard_refresh import ROOT_DIR, build_hard_refresh_steps
from src.orchestration.refresh_pipeline import build_data_refresh_steps


def test_build_data_refresh_steps_default_sequence():
    steps = build_data_refresh_steps()

    assert [step.name for step in steps] == [
        "nhl:fetch",
        "nba:fetch",
        "nhl:fetch-odds",
        "nba:fetch-odds",
    ]
    assert steps[0].command == (sys.executable, "-m", "src.cli", "fetch", "--config", "configs/nhl.yaml")
    assert steps[2].command == (sys.executable, "-m", "src.cli", "fetch-odds", "--config", "configs/nhl.yaml")


def test_build_hard_refresh_steps_default_sequence():
    steps = build_hard_refresh_steps()

    assert [step.name for step in steps] == [
        "nhl:init-db",
        "nba:init-db",
        "nhl:fetch",
        "nba:fetch",
        "nhl:fetch-odds",
        "nba:fetch-odds",
        "nhl:train",
        "nba:train",
        "staging:generate-data",
        "staging:build-pages",
    ]
    assert steps[0].command == (sys.executable, "-m", "src.cli", "init-db", "--config", "configs/nhl.yaml")
    assert steps[6].command == (sys.executable, "-m", "src.cli", "train", "--config", "configs/nhl.yaml")
    assert steps[8].cwd == ROOT_DIR / "web"
    assert all(not step.name.endswith(":features") for step in steps)


def test_build_hard_refresh_steps_models_and_approve_flag():
    steps = build_hard_refresh_steps(models_arg="glm,rf,glm", approve_feature_changes=True, include_pages_build=False)

    train_steps = [step for step in steps if step.name.endswith(":train")]
    assert len(train_steps) == 2
    for step in train_steps:
        assert step.command[-3:] == ("--models", "glm_ridge,rf", "--approve-feature-changes")

    assert [step.name for step in steps][-1] == "staging:generate-data"


def test_build_hard_refresh_steps_rejects_unknown_models():
    with pytest.raises(ValueError):
        build_hard_refresh_steps(models_arg="not_a_model")


def test_assert_publish_preconditions_requires_clean_main_worktree(monkeypatch):
    monkeypatch.setattr(hard_refresh_module, "_capture_stdout", lambda *args, **kwargs: "main")
    monkeypatch.setattr(
        hard_refresh_module,
        "_git_status_lines",
        lambda **kwargs: [" M web/public/staging-data/manifest.json"],
    )

    with pytest.raises(RuntimeError, match="clean git worktree"):
        hard_refresh_module.assert_publish_preconditions()


def test_run_publish_closeout_skips_when_no_changes(monkeypatch):
    monkeypatch.setattr(hard_refresh_module, "_git_status_lines", lambda **kwargs: [])

    assert hard_refresh_module.run_publish_closeout() is None


def test_run_publish_closeout_commits_pushes_and_watches_matching_run(monkeypatch):
    commands: list[tuple[tuple[str, ...], bool]] = []

    def fake_git_status_lines(*, root_dir=ROOT_DIR, pathspec=None):
        if pathspec == "web/public/staging-data":
            return [" M web/public/staging-data/manifest.json"]
        return [
            " M data/processed/nba_forecast.db",
            " M web/public/staging-data/manifest.json",
        ]

    def fake_run_command(command, *, cwd=ROOT_DIR, capture_output=False):
        commands.append((command, capture_output))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_capture_stdout(command, *, cwd=ROOT_DIR):
        assert command == ("git", "rev-parse", "HEAD")
        return "abc123def456"

    monkeypatch.setattr(hard_refresh_module, "_git_status_lines", fake_git_status_lines)
    monkeypatch.setattr(hard_refresh_module, "_run_command", fake_run_command)
    monkeypatch.setattr(hard_refresh_module, "_capture_stdout", fake_capture_stdout)
    monkeypatch.setattr(hard_refresh_module, "_build_commit_message", lambda: "Hard refresh 2026-03-10 12:00 CDT")
    monkeypatch.setattr(
        hard_refresh_module,
        "_wait_for_publish_workflow_run",
        lambda head_sha, *, root_dir=ROOT_DIR, attempts=12, interval_seconds=5: hard_refresh_module.WorkflowRun(
            database_id=42,
            head_sha=head_sha,
            status="queued",
            conclusion=None,
            url="https://github.com/example/actions/runs/42",
        ),
    )
    monkeypatch.setattr(
        hard_refresh_module,
        "_watch_publish_workflow",
        lambda run_id, *, root_dir=ROOT_DIR: hard_refresh_module.WorkflowRun(
            database_id=run_id,
            head_sha="abc123def456",
            status="completed",
            conclusion="success",
            url="https://github.com/example/actions/runs/42",
        ),
    )

    summary = hard_refresh_module.run_publish_closeout()

    assert summary == hard_refresh_module.HardRefreshPublishSummary(
        commit_sha="abc123def456",
        staging_data_changed=True,
        workflow_url="https://github.com/example/actions/runs/42",
        workflow_conclusion="success",
    )
    assert commands == [
        (("git", "add", "-A"), False),
        (("git", "commit", "-m", "Hard refresh 2026-03-10 12:00 CDT"), False),
        (("git", "push", "origin", "main"), False),
    ]
