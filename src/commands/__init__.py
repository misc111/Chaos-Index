"""Thin CLI command registry.

`src.cli` should only parse arguments and dispatch here. Any command that needs
substantial work belongs in a command module that delegates to a service.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from src.common.config import AppConfig
from src.registry.commands import get_command_handler

CommandHandler = Callable[[AppConfig, Namespace], None]


def dispatch(command: str, cfg: AppConfig, args: Namespace) -> None:
    """Dispatch a parsed CLI command through the canonical command registry."""

    handler = get_command_handler(command)
    typed_handler = handler if callable(handler) else None
    if typed_handler is None:
        raise KeyError(f"Unknown command '{command}'.")
    typed_handler(cfg, args)
