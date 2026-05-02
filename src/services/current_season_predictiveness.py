"""Current-season MLB predictiveness evidence service.

The public entry point in this module is intentionally thin: it coordinates
database loading, scoring, payload assembly, and artifact writes while leaving
contract details and metric construction in focused sibling modules.
"""

from __future__ import annotations

from typing import Any

from src.common.config import AppConfig
from src.common.time import utc_now_iso
from src.services.current_season_predictiveness_contract import artifact_paths, clean_payload, write_predictiveness_artifacts
from src.services.current_season_predictiveness_scoring import (
    eligible_scoring_frame,
    latest_result_season,
    prediction_rows,
    regular_result_count,
    target_payloads,
)
from src.storage.db import Database


def _season_bounds(cfg: AppConfig) -> tuple[str, str]:
    """Return the configured MLB season scoring window as inclusive ISO dates."""

    return str(cfg.data.season_start), str(cfg.data.season_end)


def run_current_season_predictiveness(
    cfg: AppConfig,
    *,
    report_slug: str | None = None,
    season: int | None = None,
    as_of_utc: str | None = None,
    windows_days: list[int] | None = None,
) -> dict[str, Any]:
    """Score frozen pregame MLB predictions against current-season outcomes.

    The returned dictionary is the exact JSON payload written to the summary
    artifact.  If the frozen ledger is empty, the service still writes a full
    ``not_measurable`` artifact so downstream dashboards and staging snapshots
    can distinguish "no live evidence yet" from a broken report generator.
    """

    db = Database(cfg.paths.db_path)
    start_date, end_date = _season_bounds(cfg)
    resolved_as_of = as_of_utc or utc_now_iso()
    resolved_season = int(season) if season is not None else latest_result_season(db)
    artifacts = artifact_paths(cfg, report_slug=report_slug)
    windows = list(windows_days if windows_days is not None else cfg.modeling.rolling_windows_days)

    raw_rows = prediction_rows(db, start_date=start_date, end_date=end_date, season=resolved_season)
    scored, row_counts = eligible_scoring_frame(raw_rows)
    target_payload, metric_rows, reliability_rows = target_payloads(scored, as_of_utc=resolved_as_of, windows_days=windows)
    status = "measurable" if not scored.empty else "not_measurable"
    report_slug_value = str(report_slug or "current_season_predictiveness").strip() or "current_season_predictiveness"
    payload = clean_payload(
        {
            "artifact_paths": artifacts.as_payload(),
            "as_of_utc": resolved_as_of,
            "betting_overlay": {
                "status": "not_measured",
                "reason": "Predictive probability skill is evaluated separately from ROI; pregame odds decisions are not part of this artifact.",
            },
            "league": str(cfg.data.league).upper(),
            "report_slug": report_slug_value,
            "scoreboard": {
                **row_counts,
                "regular_season_results": regular_result_count(
                    db,
                    start_date=start_date,
                    end_date=end_date,
                    season=resolved_season,
                ),
                "scored_live_predictions": int(len(scored)),
            },
            "season": resolved_season,
            "season_window": {
                "start_date": start_date,
                "end_date": end_date,
            },
            "status": status,
            "targets": target_payload,
        }
    )
    write_predictiveness_artifacts(
        artifacts,
        payload=payload,
        scored=scored,
        metric_rows=[clean_payload(row) for row in metric_rows],
        reliability_rows=[clean_payload(row) for row in reliability_rows],
    )
    return payload
