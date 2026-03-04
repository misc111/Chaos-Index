from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, stable_hash


class RunTracker:
    def __init__(self, root_dir: str):
        self.root = ensure_dir(Path(root_dir) / "reports" / "runs")

    def start_run(self, name: str, payload: dict[str, Any] | None = None) -> str:
        stamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
        run_id = f"{name}_{stamp}_{stable_hash(payload or {})[:8]}"
        run_dir = ensure_dir(self.root / run_id)
        meta = {
            "run_id": run_id,
            "name": name,
            "started_at_utc": utc_now_iso(),
            "payload": payload or {},
            "status": "running",
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return run_id

    def log_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        run_dir = self.root / run_id
        path = run_dir / "metrics.json"
        existing = {}
        if path.exists():
            existing = json.loads(path.read_text())
        existing.update(metrics)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True))

    def log_artifact(self, run_id: str, name: str, content: dict[str, Any] | list[Any]) -> str:
        run_dir = self.root / run_id
        path = run_dir / f"{name}.json"
        path.write_text(json.dumps(content, indent=2, default=str, sort_keys=True))
        return str(path)

    def end_run(self, run_id: str, status: str = "finished") -> None:
        run_dir = self.root / run_id
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"run_id": run_id}
        meta["ended_at_utc"] = utc_now_iso()
        meta["status"] = status
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
