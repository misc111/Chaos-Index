from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.config import AppConfig
from src.league_registry import canonicalize_league


@dataclass(frozen=True)
class ResearchPaths:
    league: str
    source_dir: Path
    source_manifest: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path


def resolve_research_paths(cfg: AppConfig) -> ResearchPaths:
    league = canonicalize_league(cfg.data.league)
    league_slug = league.lower()
    source_root = Path(cfg.research.source_dir)
    source_dir = source_root if source_root.name == league_slug else source_root / league_slug

    interim_root = Path(cfg.paths.interim_dir)
    processed_root = Path(cfg.paths.processed_dir)
    artifacts_root = Path(cfg.paths.artifacts_dir)

    return ResearchPaths(
        league=league,
        source_dir=source_dir,
        source_manifest=source_dir / cfg.research.source_manifest,
        interim_dir=interim_root.parent / "research" / league_slug,
        processed_dir=processed_root.parent / "research" / league_slug,
        artifacts_dir=artifacts_root / "research" / league_slug,
    )
