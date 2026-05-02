"""Artifact contract helpers for current-season MLB predictiveness evidence.

This module contains only stable labels, output columns, JSON normalization,
and artifact writing.  Keeping those concerns separate from SQL loading and
metric computation makes the evidence contract easy to audit and keeps the
orchestrating service small.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import AppConfig
from src.common.utils import ensure_dir, to_json


PRIMARY_TARGET = "moneyline_home_win"
RUNLINE_TARGET = "runline_home_cover"
TOTALS_TARGET = "totals_over"
TARGET_ORDER = (PRIMARY_TARGET, RUNLINE_TARGET, TOTALS_TARGET)

METRICS_COLUMNS = [
    "row_type",
    "target_name",
    "model_name",
    "window_label",
    "start_date",
    "end_date",
    "n_games",
    "log_loss",
    "brier",
    "accuracy",
    "auc",
    "ece",
    "mce",
    "calibration_alpha",
    "calibration_beta",
]
SCORES_COLUMNS = [
    "target_name",
    "game_id",
    "game_date_utc",
    "start_time_utc",
    "model_name",
    "model_run_id",
    "as_of_utc",
    "prob_home_win",
    "outcome_home_win",
]
RELIABILITY_COLUMNS = ["target_name", "model_name", "bin", "count", "pred_mean", "obs_rate", "abs_gap"]


@dataclass(frozen=True)
class CurrentSeasonPredictivenessArtifacts:
    """Filesystem outputs for one current-season predictiveness run."""

    summary_json: Path
    metrics_csv: Path
    scores_csv: Path
    reliability_csv: Path
    manifest_json: Path
    report_md: Path

    def as_payload(self) -> dict[str, str]:
        """Return JSON-safe artifact paths in a stable key order."""

        return {
            "manifest_json": str(self.manifest_json),
            "metrics_csv": str(self.metrics_csv),
            "reliability_csv": str(self.reliability_csv),
            "report_md": str(self.report_md),
            "scores_csv": str(self.scores_csv),
            "summary_json": str(self.summary_json),
        }


def artifact_paths(cfg: AppConfig, *, report_slug: str | None) -> CurrentSeasonPredictivenessArtifacts:
    """Resolve canonical MLB report paths for the evidence snapshot."""

    slug = str(report_slug or "current_season_predictiveness").strip() or "current_season_predictiveness"
    root = ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / "mlb")
    prefix = root / slug
    return CurrentSeasonPredictivenessArtifacts(
        summary_json=Path(f"{prefix}.json"),
        metrics_csv=Path(f"{prefix}.csv"),
        scores_csv=Path(f"{prefix}_scores.csv"),
        reliability_csv=Path(f"{prefix}_reliability.csv"),
        manifest_json=Path(f"{prefix}_manifest.json"),
        report_md=Path(f"{prefix}.md"),
    )


def clean_payload(value: Any) -> Any:
    """Recursively normalize a payload before deterministic JSON serialization."""

    if isinstance(value, dict):
        return {str(key): clean_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [clean_payload(item) for item in value]
    if isinstance(value, tuple):
        return [clean_payload(item) for item in value]
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, (int, str)):
        return value
    return str(value)


def write_predictiveness_artifacts(
    artifacts: CurrentSeasonPredictivenessArtifacts,
    *,
    payload: dict[str, Any],
    scored: pd.DataFrame,
    metric_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
) -> None:
    """Persist deterministic JSON, CSV, and Markdown evidence artifacts."""

    clean = clean_payload(payload)
    artifacts.summary_json.write_text(to_json(clean) + "\n")
    artifacts.manifest_json.write_text(
        to_json(
            {
                "artifact_root": str(artifacts.summary_json.parent),
                "artifacts": artifacts.as_payload(),
                "generated_at_utc": payload["as_of_utc"],
                "league": payload["league"],
                "report_slug": payload["report_slug"],
            }
        )
        + "\n"
    )

    pd.DataFrame(metric_rows, columns=METRICS_COLUMNS).to_csv(artifacts.metrics_csv, index=False)
    pd.DataFrame(reliability_rows, columns=RELIABILITY_COLUMNS).to_csv(artifacts.reliability_csv, index=False)
    scores_out = scored.copy()
    if not scores_out.empty:
        scores_out = scores_out[[column for column in SCORES_COLUMNS if column in scores_out.columns]].sort_values(
            ["target_name", "model_name", "game_date_utc", "game_id"]
        )
    pd.DataFrame(scores_out, columns=SCORES_COLUMNS).to_csv(artifacts.scores_csv, index=False)
    artifacts.report_md.write_text(render_markdown(clean) + "\n")


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact human-readable summary beside the machine artifacts."""

    scoreboard = payload["scoreboard"]
    lines = [
        "# MLB Current-Season Predictiveness",
        "",
        f"- Status: {payload['status']}",
        f"- As of UTC: {payload['as_of_utc']}",
        f"- Season window: {payload['season_window']['start_date']} to {payload['season_window']['end_date']}",
        f"- Settled regular-season results: {scoreboard['regular_season_results']}",
        f"- Scored live predictions: {scoreboard['scored_live_predictions']}",
        f"- Quarantined post-start predictions: {scoreboard['quarantined_post_start_predictions']}",
        "",
        "## Targets",
    ]
    for target_name in TARGET_ORDER:
        target = payload["targets"][target_name]
        lines.append(f"- {target_name}: {target['status']}")
    return "\n".join(lines)
