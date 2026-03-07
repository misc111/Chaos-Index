"""Progress callback helpers for long-running training stages."""

from __future__ import annotations

from typing import Any, Callable

from src.common.time import utc_now_iso

ProgressCallback = Callable[[dict[str, Any]], None]


def emit_progress(progress_callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_callback is None:
        return
    event = {"ts_utc": utc_now_iso(), **payload}
    try:
        progress_callback(event)
    except Exception:
        return
