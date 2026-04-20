from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ESPN_MLB_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"


def empty_fetch_result(source: str, *, metadata: dict[str, Any] | None = None) -> SourceFetchResult:
    extracted_at_utc = utc_now_iso()
    resolved_metadata = {"status": "scaffold", **(metadata or {})}
    return SourceFetchResult(
        source=source,
        snapshot_id=f"{source}_scaffold",
        extracted_at_utc=extracted_at_utc,
        raw_path="",
        metadata=resolved_metadata,
        dataframe=pd.DataFrame(),
    )


def fetch_game_summary(client: HttpClient, game_id: int | str) -> tuple[dict[str, Any], str]:
    event = str(game_id)
    payload, raw_path = client.get_json("mlb_summary", ESPN_MLB_SUMMARY_URL, params={"event": event}, key=event)
    return payload, raw_path


def stat_value_map(stats: list[dict[str, Any]] | None) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for stat in stats or []:
        name = str(stat.get("name") or "").strip()
        if not name:
            continue
        mapped[name] = stat.get("value", stat.get("displayValue"))
    return mapped


def athlete_stat_map(stat_block: dict[str, Any], athlete: dict[str, Any]) -> dict[str, Any]:
    keys = list(stat_block.get("keys") or [])
    if not keys and str(stat_block.get("type") or "").lower() == "pitching":
        keys = [
            "fullInnings.partInnings",
            "hits",
            "runs",
            "earnedRuns",
            "walks",
            "strikeouts",
            "homeRuns",
            "pitches-strikes",
            "ERA",
            "pitches",
        ]
    values = list(athlete.get("stats") or [])
    return {str(key): values[index] if index < len(values) else None for index, key in enumerate(keys)}


def parse_baseball_innings(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        if "." not in raw:
            return float(raw)
        whole, partial = raw.split(".", 1)
        outs = int(partial[:1] or "0")
        return int(whole) + outs / 3.0
    except Exception:
        return None
