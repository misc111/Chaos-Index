"""Canonical CLI command registry shared by argparse, docs, and help text."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from src.registry.command_definitions import COMMAND_REGISTRY
from src.registry.leagues import default_config_path
from src.registry.types import CommandRegistryEntry

_COMMAND_BY_NAME = {entry.name: entry for entry in COMMAND_REGISTRY}


def command_registry() -> tuple[CommandRegistryEntry, ...]:
    """Return all canonical CLI command definitions."""

    return COMMAND_REGISTRY


def command_names() -> tuple[str, ...]:
    """Return the canonical CLI command name tuple."""

    return tuple(entry.name for entry in COMMAND_REGISTRY)


def get_command_spec(name: str) -> CommandRegistryEntry:
    """Resolve a command name into registry metadata."""

    return _COMMAND_BY_NAME[name]


def get_command_handler(name: str) -> Callable[..., object]:
    """Resolve the callable handler for a registered command."""

    spec = get_command_spec(name)
    module_name, attribute_name = spec.handler_path.split(":", 1)
    module = importlib.import_module(module_name)
    return cast(Callable[..., object], getattr(module, attribute_name))


def command_manifest_payload() -> dict[str, object]:
    """Render the deterministic command manifest payload."""

    return {
        "version": 1,
        "source": "code_registry",
        "default_config_path": default_config_path("MLB"),
        "commands": [
            {
                "name": entry.name,
                "summary": entry.summary,
                "handler_path": entry.handler_path,
                "config_required": entry.config_required,
                "arguments": [
                    {
                        "flags": list(argument.flags),
                        "doc_metavar": argument.doc_metavar,
                        "help": argument.argparse_kwargs.get("help"),
                        "choices": list(argument.argparse_kwargs["choices"])
                        if "choices" in argument.argparse_kwargs
                        else None,
                        "default": argument.argparse_kwargs.get("default"),
                    }
                    for argument in entry.arguments
                ],
                "examples": list(entry.examples),
            }
            for entry in COMMAND_REGISTRY
        ],
    }


def render_make_help() -> str:
    """Render the dynamic CLI section for `make help`."""

    lines = [
        "CLI-backed targets:",
    ]
    for entry in COMMAND_REGISTRY:
        lines.append(f"  {entry.name:<18} {entry.summary}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_make_help())
