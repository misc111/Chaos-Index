from __future__ import annotations

from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.mlb._helpers import empty_fetch_result


def fetch_weather_context_optional(
    client: HttpClient,
    *args: object,
    **kwargs: object,
) -> SourceFetchResult:
    del client, args, kwargs
    return empty_fetch_result(
        "mlb_weather",
        metadata={
            "note": "Public ESPN scoreboard payload does not expose stable pregame weather for MLB in this lane yet",
        },
    )
