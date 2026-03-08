"""Thin CLI command registry.

`src.cli` should only parse arguments and dispatch here. Any command that needs
substantial work belongs in a command module that delegates to a service.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from src.common.config import AppConfig

from . import data, modeling, smoke

CommandHandler = Callable[[AppConfig, Namespace], None]


def dispatch(command: str, cfg: AppConfig, args: Namespace) -> None:
    handlers: dict[str, CommandHandler] = {
        "init-db": data.init_db,
        "fetch": data.fetch,
        "refresh-data": data.refresh_data,
        "fetch-odds": data.fetch_odds,
        "features": data.features,
        "research-features": data.research_features,
        "train": modeling.train,
        "validate": modeling.validate,
        "backtest": modeling.backtest,
        "run-daily": modeling.run_daily,
        "smoke": smoke.run,
    }
    handlers[command](cfg, args)
