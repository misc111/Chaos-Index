"""Shared access to generated manifests consumed by Python and the web app.

These files are committed artifacts so orchestration code can depend on one
canonical source of league and model metadata instead of duplicating lists.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast


ROOT_DIR = Path(__file__).resolve().parents[2]
GENERATED_CONFIG_DIR = ROOT_DIR / "configs" / "generated"


def generated_config_path(file_name: str) -> Path:
    """Return the absolute path to a generated manifest file."""

    return GENERATED_CONFIG_DIR / file_name


def _load_json(file_name: str) -> dict[str, Any]:
    path = generated_config_path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Generated manifest not found: {path}")
    return cast(dict[str, Any], json.loads(path.read_text()))


@lru_cache(maxsize=1)
def load_league_manifest() -> dict[str, Any]:
    """Load the generated league manifest."""

    return _load_json("league_manifest.json")


@lru_cache(maxsize=1)
def load_model_manifest() -> dict[str, Any]:
    """Load the generated model manifest."""

    return _load_json("model_manifest.json")


@lru_cache(maxsize=1)
def load_command_manifest() -> dict[str, Any]:
    """Load the generated command manifest."""

    return _load_json("command_manifest.json")


@lru_cache(maxsize=1)
def load_dashboard_route_manifest() -> dict[str, Any]:
    """Load the generated dashboard route manifest."""

    return _load_json("dashboard_route_manifest.json")
