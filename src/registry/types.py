"""Typed registry primitives for code-first repo metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LeagueRegistryEntry:
    """Canonical runtime metadata for a supported league."""

    code: str
    slug: str
    display_label: str
    default_config_path: str
    config_env_var: str
    project_name: str
    db_path: str
    db_env_var: str
    championship_name: str
    championship_probability_key: str
    uncertainty_policy_name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Canonical metadata for a trainable or reportable model."""

    key: str
    display_label: str
    short_label: str
    family: str
    aliases: tuple[str, ...] = ()
    legacy_model_keys: tuple[str, ...] = ()
    trainable: bool = True
    prediction_report_rank: int = 0


@dataclass(frozen=True)
class CommandArgumentSpec:
    """Declarative argparse metadata for a command-line option."""

    flags: tuple[str, ...]
    argparse_kwargs: dict[str, Any] = field(default_factory=dict)
    doc_metavar: str | None = None


@dataclass(frozen=True)
class CommandRegistryEntry:
    """Canonical CLI command definition."""

    name: str
    summary: str
    handler_path: str
    arguments: tuple[CommandArgumentSpec, ...] = ()
    examples: tuple[str, ...] = ()
    config_required: bool = True


@dataclass(frozen=True)
class DashboardRouteRegistryEntry:
    """Canonical metadata for a dashboard API/staging route."""

    key: str
    summary: str
    module_path: str
    api_path: str
    staging_file_name: str
    payload_contract: str
    page_path: str | None = None
    include_in_staging: bool = True
    public: bool = True
    supports_experiments: bool = False


@dataclass(frozen=True)
class SubsystemDocEntry:
    """Canonical documentation metadata for a repo subsystem."""

    path: str
    title: str
    summary: str
    public_entrypoints: tuple[str, ...]
    readme_path: str | None = None
    generate_readme: bool = False
    notes: tuple[str, ...] = ()
