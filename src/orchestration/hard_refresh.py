from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.training.train import normalize_selected_models


ROOT_DIR = Path(__file__).resolve().parents[2]
LEAGUE_CONFIGS: tuple[tuple[str, str], ...] = (
    ("nhl", "configs/nhl.yaml"),
    ("nba", "configs/nba.yaml"),
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


def build_hard_refresh_steps(
    *,
    root_dir: Path | None = None,
    models_arg: str | None = None,
    approve_feature_changes: bool = False,
    include_pages_build: bool = True,
) -> list[OrchestrationStep]:
    resolved_root = ROOT_DIR if root_dir is None else Path(root_dir).resolve()
    models_csv = _normalize_models_arg(models_arg)
    steps: list[OrchestrationStep] = []

    for league, config_path in LEAGUE_CONFIGS:
        steps.extend(
            [
                OrchestrationStep(
                    name=f"{league}:init-db",
                    command=_cli_command("init-db", "--config", config_path),
                    cwd=resolved_root,
                ),
                OrchestrationStep(
                    name=f"{league}:fetch",
                    command=_cli_command("fetch", "--config", config_path),
                    cwd=resolved_root,
                ),
                OrchestrationStep(
                    name=f"{league}:fetch-odds",
                    command=_cli_command("fetch-odds", "--config", config_path),
                    cwd=resolved_root,
                ),
                OrchestrationStep(
                    name=f"{league}:features",
                    command=_cli_command("features", "--config", config_path),
                    cwd=resolved_root,
                ),
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
                ),
            ]
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


def run_steps(steps: Sequence[OrchestrationStep]) -> None:
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{total}] {step.name}", flush=True)
        print(f"  cwd={step.cwd}", flush=True)
        print(f"  cmd={step.display_command}", flush=True)
        try:
            subprocess.run(step.command, cwd=step.cwd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Hard refresh failed at step '{step.name}' with exit code {exc.returncode}: {step.display_command}"
            ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic full hard-refresh pipeline across NHL and NBA.")
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model list to train for both leagues. Defaults to the full model suite.",
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
        return

    run_steps(steps)


if __name__ == "__main__":
    main()
