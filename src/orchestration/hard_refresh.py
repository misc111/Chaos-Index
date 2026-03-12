"""CLI entry point for the deterministic repo-wide hard refresh pipeline."""

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.orchestration.refresh_pipeline import ROOT_DIR, build_hard_refresh_steps, run_steps


PUBLISH_WORKFLOW_NAME = "Publish Sanitized Staging Site"


@dataclass(frozen=True)
class WorkflowRun:
    """GitHub Actions workflow metadata for the publish workflow."""

    database_id: int
    head_sha: str
    status: str
    conclusion: str | None
    url: str


@dataclass(frozen=True)
class HardRefreshPublishSummary:
    """Summary of the commit/push/workflow closeout after a hard refresh."""

    commit_sha: str
    staging_data_changed: bool
    workflow_url: str
    workflow_conclusion: str | None


def _run_command(
    command: tuple[str, ...],
    *,
    cwd: Path = ROOT_DIR,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Hard refresh publish closeout failed with exit code {exc.returncode}: {shlex.join(command)}"
        ) from exc


def _capture_stdout(command: tuple[str, ...], *, cwd: Path = ROOT_DIR) -> str:
    return _run_command(command, cwd=cwd, capture_output=True).stdout.strip()


def _git_status_lines(*, root_dir: Path = ROOT_DIR, pathspec: str | None = None) -> list[str]:
    command = ["git", "status", "--short", "--untracked-files=all"]
    if pathspec is not None:
        command.extend(["--", pathspec])
    output = _capture_stdout(tuple(command), cwd=root_dir)
    return [line for line in output.splitlines() if line.strip()]


def assert_publish_preconditions(*, root_dir: Path = ROOT_DIR) -> None:
    """Require a clean `main` worktree before a hard refresh publish closeout."""

    current_branch = _capture_stdout(("git", "branch", "--show-current"), cwd=root_dir)
    if current_branch != "main":
        raise RuntimeError(
            f"Hard refresh publish closeout requires branch 'main'; current branch is '{current_branch}'."
        )

    dirty_lines = _git_status_lines(root_dir=root_dir)
    if dirty_lines:
        dirty_preview = "\n".join(dirty_lines[:10])
        raise RuntimeError(
            "Hard refresh publish closeout requires a clean git worktree before execution so the automated commit "
            "only includes refresh artifacts. Clean or stash the existing changes and rerun.\n"
            f"{dirty_preview}"
        )


def _list_publish_workflow_runs(*, root_dir: Path = ROOT_DIR) -> list[WorkflowRun]:
    raw_payload = _capture_stdout(
        (
            "gh",
            "run",
            "list",
            "--workflow",
            PUBLISH_WORKFLOW_NAME,
            "--limit",
            "5",
            "--json",
            "databaseId,headSha,status,conclusion,url,displayTitle",
        ),
        cwd=root_dir,
    )
    payload = json.loads(raw_payload or "[]")
    runs: list[WorkflowRun] = []
    for item in payload:
        runs.append(
            WorkflowRun(
                database_id=int(item["databaseId"]),
                head_sha=str(item["headSha"]),
                status=str(item["status"]),
                conclusion=item.get("conclusion"),
                url=str(item["url"]),
            )
        )
    return runs


def _wait_for_publish_workflow_run(
    head_sha: str,
    *,
    root_dir: Path = ROOT_DIR,
    attempts: int = 12,
    interval_seconds: int = 5,
) -> WorkflowRun:
    for attempt in range(1, attempts + 1):
        for workflow_run in _list_publish_workflow_runs(root_dir=root_dir):
            if workflow_run.head_sha == head_sha:
                return workflow_run
        if attempt < attempts:
            print(
                f"Waiting for '{PUBLISH_WORKFLOW_NAME}' workflow to register for commit {head_sha[:7]}...",
                flush=True,
            )
            time.sleep(interval_seconds)
    raise RuntimeError(
        f"Unable to find a '{PUBLISH_WORKFLOW_NAME}' GitHub Actions run for pushed commit {head_sha}."
    )


def _view_workflow_run(run_id: int, *, root_dir: Path = ROOT_DIR) -> WorkflowRun:
    raw_payload = _capture_stdout(
        ("gh", "run", "view", str(run_id), "--json", "databaseId,headSha,status,conclusion,url"),
        cwd=root_dir,
    )
    item = json.loads(raw_payload or "{}")
    return WorkflowRun(
        database_id=int(item["databaseId"]),
        head_sha=str(item["headSha"]),
        status=str(item["status"]),
        conclusion=item.get("conclusion"),
        url=str(item["url"]),
    )


def _watch_publish_workflow(run_id: int, *, root_dir: Path = ROOT_DIR) -> WorkflowRun:
    watch_command = ("gh", "run", "watch", str(run_id), "--interval", "5")
    watch_failed = False
    try:
        subprocess.run(watch_command, cwd=root_dir, check=True, text=True)
    except subprocess.CalledProcessError:
        watch_failed = True

    workflow_run = _view_workflow_run(run_id, root_dir=root_dir)
    if watch_failed and workflow_run.conclusion is None:
        raise RuntimeError(
            f"Hard refresh publish closeout could not determine the final state of workflow run {run_id}: "
            f"{shlex.join(watch_command)}"
        )
    return workflow_run


def _build_commit_message() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"Hard refresh {timestamp}"


def run_publish_closeout(*, root_dir: Path = ROOT_DIR) -> HardRefreshPublishSummary | None:
    """Commit, push, and watch the publish workflow for a successful hard refresh."""

    changed_lines = _git_status_lines(root_dir=root_dir)
    if not changed_lines:
        print("Hard refresh produced no repository changes; skipping git commit, push, and workflow watch.", flush=True)
        return None

    staging_data_changed = bool(_git_status_lines(root_dir=root_dir, pathspec="web/public/staging-data"))
    _run_command(("git", "add", "-A"), cwd=root_dir)
    _run_command(("git", "commit", "-m", _build_commit_message()), cwd=root_dir)
    commit_sha = _capture_stdout(("git", "rev-parse", "HEAD"), cwd=root_dir)
    _run_command(("git", "push", "origin", "main"), cwd=root_dir)
    workflow_run = _wait_for_publish_workflow_run(commit_sha, root_dir=root_dir)
    workflow_run = _watch_publish_workflow(workflow_run.database_id, root_dir=root_dir)

    summary = HardRefreshPublishSummary(
        commit_sha=commit_sha,
        staging_data_changed=staging_data_changed,
        workflow_url=workflow_run.url,
        workflow_conclusion=workflow_run.conclusion,
    )

    print(
        f"Pushed {summary.commit_sha}; staging-data changed={'yes' if summary.staging_data_changed else 'no'}; "
        f"workflow={summary.workflow_url}; conclusion={summary.workflow_conclusion or 'unknown'}",
        flush=True,
    )

    if summary.workflow_conclusion != "success":
        raise RuntimeError(
            f"Hard refresh pushed commit {summary.commit_sha}, but workflow '{PUBLISH_WORKFLOW_NAME}' "
            f"finished with conclusion '{summary.workflow_conclusion or 'unknown'}': {summary.workflow_url}"
        )

    return summary


def main() -> None:
    """Run or preview the deterministic repo-wide hard refresh pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the deterministic multi-league hard-refresh pipeline without rebuilding features."
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model list to train for all supported leagues. Defaults to the full model suite.",
    )
    parser.add_argument(
        "--approve-feature-changes",
        action="store_true",
        help="Explicitly allow feature-contract updates during the train step.",
    )
    parser.add_argument(
        "--skip-pages-build",
        action="store_true",
        help="Skip `npm run build:pages` after generating staging data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the hard-refresh step plan without executing it.",
    )
    args = parser.parse_args()

    steps = build_hard_refresh_steps(
        models_arg=args.models,
        approve_feature_changes=bool(args.approve_feature_changes),
        include_pages_build=not bool(args.skip_pages_build),
    )

    if args.dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index}/{len(steps)}] {step.name}")
            print(f"  cwd={step.cwd}")
            print(f"  cmd={step.display_command}")
        publish_start = len(steps) + 1
        print(f"[{publish_start}/{len(steps) + 3}] git:commit-refresh-results")
        print(f"  cwd={ROOT_DIR}")
        print("  cmd=git add -A && git commit -m 'Hard refresh <timestamp>'")
        print(f"[{publish_start + 1}/{len(steps) + 3}] git:push-main")
        print(f"  cwd={ROOT_DIR}")
        print("  cmd=git push origin main")
        print(f"[{publish_start + 2}/{len(steps) + 3}] gh:watch-pages-publish")
        print(f"  cwd={ROOT_DIR}")
        print(
            "  cmd=gh run list --workflow 'Publish Sanitized Staging Site' ... && gh run watch <databaseId> --interval 5"
        )
        return

    assert_publish_preconditions()
    run_steps(steps, pipeline_name="Hard refresh")
    run_publish_closeout()


if __name__ == "__main__":
    main()
