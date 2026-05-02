from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

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
        self._json_cache: dict[tuple[str, str, str, str], tuple[Any, str]] = {}
        self._json_header_cache: dict[tuple[str, str, str, str, str], tuple[Any, str, dict[str, str], bool]] = {}

    def _source_dir(self, source: str) -> Path:
        date_stamp = utc_now_iso()[:10]
        return ensure_dir(self.raw_dir / source / date_stamp)

    def save_raw(self, source: str, payload: Any, key: str = "snapshot") -> str:
        path = self._source_dir(source) / f"{key}_{int(time.time())}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

    def latest_cached_file(self, source: str, key: str | None = None) -> Path | None:
        source_root = self.raw_dir / source
        if not source_root.exists():
            return None
        pattern = "**/*.json" if key is None else f"**/{key}_*.json"
        candidates = sorted(source_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def load_latest_cached(self, source: str, key: str | None = None) -> Any | None:
        file_path = self.latest_cached_file(source, key=key)
        if not file_path:
            return None
        logger.info("Using cached %s payload from %s", source, file_path)
        return json.loads(file_path.read_text())

    def _request_retrying(self) -> Retrying:
        attempts = max(1, int(self.max_retries))
        backoff = max(0.0, float(self.backoff_seconds))
        return Retrying(
            retry=retry_if_exception_type(requests.RequestException),
            wait=wait_exponential(multiplier=backoff or 1, min=backoff, max=max(20.0, backoff)),
            stop=stop_after_attempt(attempts),
            reraise=True,
        )

    def _request_response(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        for attempt in self._request_retrying():
            with attempt:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response
        raise RuntimeError("unreachable retry state")

    def _request(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self._request_response(url, params=params)
        return response.json()

    def _cache_key(self, source: str, url: str, params: dict[str, Any] | None, key: str) -> tuple[str, str, str, str]:
        params_key = json.dumps(params or {}, sort_keys=True, default=str)
        return (source, url, params_key, key)

    def _header_cache_key(
        self,
        source: str,
        url: str,
        params: dict[str, Any] | None,
        key: str,
        headers: dict[str, str] | None,
    ) -> tuple[str, str, str, str, str]:
        headers_key = json.dumps(headers or {}, sort_keys=True, default=str)
        return (*self._cache_key(source, url, params, key), headers_key)

    def get_json(self, source: str, url: str, params: dict[str, Any] | None = None, key: str = "snapshot") -> tuple[Any, str]:
        cache_key = self._cache_key(source, url, params, key)
        if cache_key in self._json_cache:
            return self._json_cache[cache_key]

        if self.offline_mode:
            cached = self.load_latest_cached(source, key=key)
            if cached is None:
                raise RuntimeError(f"offline_mode=True but no cache available for source={source} key={key}")
            cached_file = self.latest_cached_file(source, key=key)
            result = (cached, str(cached_file) if cached_file else "")
            self._json_cache[cache_key] = result
            return result

        try:
            payload = self._request(url, params=params)
            raw_path = self.save_raw(source, payload, key=key)
            result = (payload, raw_path)
            self._json_cache[cache_key] = result
            return result
        except Exception as exc:
            logger.warning("Source %s failed live fetch (%s); attempting cache fallback", source, exc)
            cached = self.load_latest_cached(source, key=key)
            if cached is None:
                raise
            cached_file = self.latest_cached_file(source, key=key)
            result = (cached, str(cached_file) if cached_file else "")
            self._json_cache[cache_key] = result
            return result

    def get_json_with_headers(
        self,
        source: str,
        url: str,
        params: dict[str, Any] | None = None,
        key: str = "snapshot",
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, str, dict[str, str], bool]:
        cache_key = self._header_cache_key(source, url, params, key, headers)
        if cache_key in self._json_header_cache:
            return self._json_header_cache[cache_key]

        if self.offline_mode:
            cached = self.load_latest_cached(source, key=key)
            if cached is None:
                raise RuntimeError(f"offline_mode=True but no cache available for source={source} key={key}")
            cached_file = self.latest_cached_file(source, key=key)
            result = (cached, str(cached_file) if cached_file else "", {}, True)
            self._json_header_cache[cache_key] = result
            return result

        try:
            response = self._request_response(url, params=params, headers=headers)
            payload = response.json()
            raw_path = self.save_raw(source, payload, key=key)
            result = (payload, raw_path, dict(response.headers), False)
            self._json_header_cache[cache_key] = result
            return result
        except Exception as exc:
            logger.warning("Source %s failed live fetch (%s); attempting cache fallback", source, exc)
            cached = self.load_latest_cached(source, key=key)
            if cached is None:
                raise
            cached_file = self.latest_cached_file(source, key=key)
            result = (cached, str(cached_file) if cached_file else "", {}, True)
            self._json_header_cache[cache_key] = result
            return result

    def snapshot_id(self, source: str, metadata: dict[str, Any]) -> str:
        return f"{source}_{stable_hash(metadata)}"
