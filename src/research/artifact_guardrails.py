"""Write-path guardrails for MLB primary research artifacts."""

from __future__ import annotations

from pathlib import Path


PRIMARY_ARTIFACT_LEAGUE = "mlb"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def require_mlb_report_path(path: Path, artifacts_dir: str | Path, *, purpose: str) -> Path:
    """Require an MLB report path to stay under artifacts/reports/mlb."""

    candidate = Path(path).expanduser().resolve()
    root = (Path(artifacts_dir).expanduser() / "reports" / PRIMARY_ARTIFACT_LEAGUE).resolve()
    if not _is_relative_to(candidate, root):
        raise RuntimeError(
            f"{purpose} must write under artifacts/reports/mlb, got {candidate}. "
            "Use legacy lanes only for explicit non-MLB compatibility work."
        )
    return Path(path)


def require_mlb_tournament_path(path: Path, artifacts_dir: str | Path, *, purpose: str) -> Path:
    """Require an MLB tournament path to stay under artifacts/reports/mlb/tournament."""

    candidate = Path(path).expanduser().resolve()
    root = (Path(artifacts_dir).expanduser() / "reports" / PRIMARY_ARTIFACT_LEAGUE / "tournament").resolve()
    if not _is_relative_to(candidate, root):
        raise RuntimeError(f"{purpose} must write under artifacts/reports/mlb/tournament, got {candidate}.")
    return Path(path)
