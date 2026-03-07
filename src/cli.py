"""Application-layer CLI entry point.

This module only owns argument parsing, environment bootstrap, and command
dispatch. Command behavior lives in `src.commands` and the underlying services.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

from sklearn.exceptions import ConvergenceWarning

from src.commands import dispatch
from src.common.config import load_config
from src.common.logging import setup_logging


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        env_key = key.strip()
        if not env_key or env_key in os.environ:
            continue
        env_value = value.strip()
        if len(env_value) >= 2 and env_value[0] == env_value[-1] and env_value[0] in {"'", '"'}:
            env_value = env_value[1:-1]
        os.environ[env_key] = env_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NHL/NBA probabilistic forecasting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in [
        "init-db",
        "fetch",
        "refresh-data",
        "fetch-odds",
        "features",
        "research-features",
        "train",
        "backtest",
        "run-daily",
        "smoke",
    ]:
        p = sub.add_parser(cmd)
        p.add_argument("--config", default="configs/nhl.yaml")
        if cmd in {"research-features", "train", "backtest", "run-daily"}:
            p.add_argument(
                "--models",
                default="all",
                help="Comma-separated model list (e.g. glm_ridge,rf) or 'all'",
            )
            p.add_argument(
                "--approve-feature-changes",
                action="store_true",
                help="Explicitly accept and persist model feature-contract changes in the registry.",
            )
    return parser


def main() -> None:
    _load_dotenv_file(Path(".env"))
    _load_dotenv_file(Path("web/.env.local"))

    args = build_parser().parse_args()
    cfg = load_config(args.config)
    setup_logging("INFO")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="X has feature names")
    dispatch(args.command, cfg, args)


if __name__ == "__main__":
    main()
