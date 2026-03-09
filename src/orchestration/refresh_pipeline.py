from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.league_registry import get_league_metadata
from src.training.train import normalize_selected_models


ROOT_DIR = Path(__file__).resolve().parents[2]
LEAGUE_CONFIGS: tuple[tuple[str, str], ...] = tuple(
    (get_league_metadata(code).slug, get_league_metadata(code).default_config_path)
    for code in ("NHL", "NBA")
)


@dataclass(frozen=True)
class OrchestrationStep:
    name: str
    command: tuple[str, ...]
    cwd: Path

    @property
    def display_command(self) -> str:
        return shlex.join(self.command)


def _normalize_models_arg(models_arg: str | None) -> str | None:
    if models_arg is None:
        return None
    tokens = [token.strip() for token in str(models_arg).split(",") if token.strip()]
    if not tokens:
        return None
    return ",".join(normalize_selected_models(tokens))


def _cli_command(
    *parts: str,
    models_csv: str | None = None,
    approve_feature_changes: bool = False,
) -> tuple[str, ...]:
    command = [sys.executable, "-m", "src.cli", *parts]
    if models_csv is not None:
        command.extend(["--models", models_csv])
    if approve_feature_changes:
        command.append("--approve-feature-changes")
    return tuple(command)


def build_data_refresh_steps(*, root_dir: Path | None = None, include_init_db: bool = False) -> list[OrchestrationStep]:
    resolved_root = ROOT_DIR if root_dir is None else Path(root_dir).resolve()
    steps: list[OrchestrationStep] = []

    if include_init_db:
        for league, config_path in LEAGUE_CONFIGS:
            steps.append(
                OrchestrationStep(
                    name=f"{league}:init-db",
                    command=_cli_command("init-db", "--config", config_path),
                    cwd=resolved_root,
                )
            )

    for league, config_path in LEAGUE_CONFIGS:
        steps.append(
            OrchestrationStep(
                name=f"{league}:fetch",
                command=_cli_command("fetch", "--config", config_path),
                cwd=resolved_root,
            )
        )

    for league, config_path in LEAGUE_CONFIGS:
        steps.append(
            OrchestrationStep(
                name=f"{league}:fetch-odds",
                command=_cli_command("fetch-odds", "--config", config_path),
                cwd=resolved_root,
            )
        )

    return steps


def build_hard_refresh_steps(
    *,
    root_dir: Path | None = None,
    models_arg: str | None = None,
    approve_feature_changes: bool = False,
    include_pages_build: bool = True,
) -> list[OrchestrationStep]:
    resolved_root = ROOT_DIR if root_dir is None else Path(root_dir).resolve()
    models_csv = _normalize_models_arg(models_arg)
    steps = build_data_refresh_steps(root_dir=resolved_root, include_init_db=True)

    for league, config_path in LEAGUE_CONFIGS:
        steps.append(
            OrchestrationStep(
                name=f"{league}:train",
                command=_cli_command(
                    "train",
                    "--config",
                    config_path,
                    models_csv=models_csv,
                    approve_feature_changes=approve_feature_changes,
                ),
                cwd=resolved_root,
            )
        )

    web_dir = resolved_root / "web"
    steps.append(
        OrchestrationStep(
            name="staging:generate-data",
            command=("npm", "run", "generate:staging-data"),
            cwd=web_dir,
        )
    )
    if include_pages_build:
        steps.append(
            OrchestrationStep(
                name="staging:build-pages",
                command=("npm", "run", "build:pages"),
                cwd=web_dir,
            )
        )
    return steps


def run_steps(steps: Sequence[OrchestrationStep], *, pipeline_name: str) -> None:
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{total}] {step.name}", flush=True)
        print(f"  cwd={step.cwd}", flush=True)
        print(f"  cmd={step.display_command}", flush=True)
        try:
            subprocess.run(step.command, cwd=step.cwd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{pipeline_name} failed at step '{step.name}' with exit code {exc.returncode}: {step.display_command}"
            ) from exc
