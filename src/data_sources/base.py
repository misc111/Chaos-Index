from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.common.logging import get_logger
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, stable_hash

logger = get_logger(__name__)


@dataclass
class SourceFetchResult:
    source: str
    snapshot_id: str
    extracted_at_utc: str
    raw_path: str
    metadata: dict[str, Any]
    dataframe: pd.DataFrame


class HttpClient:
    def __init__(
        self,
        raw_dir: str,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        offline_mode: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.offline_mode = offline_mode

    def _source_dir(self, source: str) -> Path:
        date_stamp = utc_now_iso()[:10]
        return ensure_dir(self.raw_dir / source / date_stamp)

    def save_raw(self, source: str, payload: Any, key: str = "snapshot") -> str:
        path = self._source_dir(source) / f"{key}_{int(time.time())}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

    def latest_cached_file(self, source: str) -> Path | None:
        source_root = self.raw_dir / source
        if not source_root.exists():
            return None
        candidates = sorted(source_root.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def load_latest_cached(self, source: str) -> Any | None:
        file_path = self.latest_cached_file(source)
        if not file_path:
            return None
        logger.info("Using cached %s payload from %s", source, file_path)
        return json.loads(file_path.read_text())

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request_response(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        response = requests.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self._request_response(url, params=params)
        return response.json()

    def get_json(self, source: str, url: str, params: dict[str, Any] | None = None, key: str = "snapshot") -> tuple[Any, str]:
        if self.offline_mode:
            cached = self.load_latest_cached(source)
            if cached is None:
                raise RuntimeError(f"offline_mode=True but no cache available for source={source}")
            return cached, str(self.latest_cached_file(source))

        try:
            payload = self._request(url, params=params)
            raw_path = self.save_raw(source, payload, key=key)
            return payload, raw_path
        except Exception as exc:
            logger.warning("Source %s failed live fetch (%s); attempting cache fallback", source, exc)
            cached = self.load_latest_cached(source)
            if cached is None:
                raise
            return cached, str(self.latest_cached_file(source))

    def get_json_with_headers(
        self,
        source: str,
        url: str,
        params: dict[str, Any] | None = None,
        key: str = "snapshot",
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, str, dict[str, str], bool]:
        if self.offline_mode:
            cached = self.load_latest_cached(source)
            if cached is None:
                raise RuntimeError(f"offline_mode=True but no cache available for source={source}")
            cached_file = self.latest_cached_file(source)
            return cached, str(cached_file) if cached_file else "", {}, True

        try:
            response = self._request_response(url, params=params, headers=headers)
            payload = response.json()
            raw_path = self.save_raw(source, payload, key=key)
            return payload, raw_path, dict(response.headers), False
        except Exception as exc:
            logger.warning("Source %s failed live fetch (%s); attempting cache fallback", source, exc)
            cached = self.load_latest_cached(source)
            if cached is None:
                raise
            cached_file = self.latest_cached_file(source)
            return cached, str(cached_file) if cached_file else "", {}, True

    def snapshot_id(self, source: str, metadata: dict[str, Any]) -> str:
        return f"{source}_{stable_hash(metadata)}"
